import argparse
import os
import sys
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from utils import *
from network.RCF.models import RCF
import time


def strToBool(str):
    return str.lower() in ('true', 'yes', 'on', 't', '1')
parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)
parser.add_argument("--out", type=str, default='results', help='path of save results')
parser.add_argument("--testset",  default='raindrop', const='raindrop', nargs='?', choices=['raindrop'], help='')
parser.add_argument("--model",  default='all', const='all', nargs='?', choices=['all',
                                                                                'U_D_raindrop',
                                                                                'UnfairGAN_raindrop',
                                                                                'CycleGAN',], help='')
parser.add_argument("--save_img", type='bool', default=True, help='')
parser.add_argument("--debug", type='bool', default=False, help='')
parser.add_argument("--gpu", type='bool', default=True, help='')
parser.set_defaults(feature=True)

opt = parser.parse_args()
print(opt)

if opt.model == 'all':
    dicts = [
        'rain',
        'CycleGAN',
        'UnfairGAN_raindrop',
    ]
else:
    dicts = [
        opt.model,
        ]
testsets = [opt.testset]

## GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.gpu)else "cpu")

results = {}

sys.stdout.write('> Run ...')
with torch.no_grad():
    for r in dicts:
        results[r] = {}
        from network.unfairGan import Generator
        # estNet
        estNet = Generator(inRM_chs=0, inED_chs=0, mainblock_type='U_D', act_type='ReLU', nfeats=16).to(device)
        estNet = nn.DataParallel(estNet, device_ids=[device])
        estNet.load_state_dict(torch.load('weight/U_D_raindrop.pth',map_location=device))
        estNet.eval()
        # rcf net
        rcfNet = RCF().to(device)
        rcfNet.load_state_dict(torch.load('weight/RCF.pth',map_location=device)['state_dict'])
        rcfNet.eval()
        # unfairGAN net
        if 'UnfairGAN_raindrop' in r:
            unfairGAN = Generator(inRM_chs=1, inED_chs=3, mainblock_type='U_D', act_type='DAF', nfeats=16).to(device)
            unfairGAN = nn.DataParallel(unfairGAN, device_ids=[device])
            unfairGAN.load_state_dict(torch.load('weight/%s.pth' % r,map_location=device))
            unfairGAN.eval()
        # CycleGAN
        if 'CycleGAN' in r:
            from network.CycleGAN.models import networks
            from network.CycleGAN.util import util 
            CycleGAN = networks.define_G(3, 3, 64, 'resnet_9blocks','instance', not True, 'normal', 0.02, [0])
            if isinstance(CycleGAN, torch.nn.DataParallel):
                CycleGAN = CycleGAN.module
            CycleGAN.load_state_dict(torch.load('weight/%s_raindrop.pth' % r,map_location=device))
            CycleGAN.eval()

        for testset in testsets:
            ls = os.listdir('testsets/%s/rain' % testset)
            print(testset, len(ls))
            results[r][testset] = {'psnr': [], 'ssim': [], 'time': []}
            for i, img in enumerate(ls):
                if opt.debug and i > 0: continue
                # input
                input_cv2 = cv2.imread(os.path.join('testsets/%s/rain' % testset, img))
                input = align_to_num(input_cv2)
                input = to_tensor(input, device)
                # input align to 16
                input_a16 = align_to_num(input_cv2, 16)
                input_a16 = to_tensor(input_a16, device)
                # target
                target_cv2 = cv2.imread(os.path.join('testsets/%s/gt' % testset, img.replace('rain', 'clean')))
                target = align_to_num(target_cv2)
                target = to_tensor(target, device)
                # target align to 16
                target_a16 = align_to_num(target_cv2, 16)
                target_a16 = to_tensor(target_a16, device)
                # initial for measurement
                cal_input = input
                cal_target = target
                start_time = time.time()
                # rainmap
                est = estNet(input)
                logimg = make_grid(est.data.clamp(0., 1.), nrow=8, normalize=False, scale_each=False)
                est = logimg.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[:, :, ::-1]
                derain = align_to_num(est)
                rainmap = make_rainmap(input_cv2, derain)
                rainmap = to_tensor(rainmap, device)
                # edge
                derain = prepare_image_cv2(np.array(est, dtype=np.float32))
                derain_in = derain.transpose((1, 2, 0))
                scale = [0.5, 1, 1.5]
                _, H, W = derain.shape
                multi_fuse = np.zeros((H, W), np.float32)
                for k in range(0, len(scale)):
                    im_ = cv2.resize(derain_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    im_ = im_.transpose((2, 0, 1))
                    edges = rcfNet(torch.unsqueeze(torch.from_numpy(im_).to(device), 0))
                    edge = torch.squeeze(edges[-1].detach()).cpu().numpy()
                    fuse = cv2.resize(edge, (W, H), interpolation=cv2.INTER_LINEAR)
                    multi_fuse += fuse
                multi_fuse = multi_fuse / len(scale)
                edge = (multi_fuse * 255).astype(np.uint8)
                edge = np.stack([edge,edge,edge])
                edge = np.array(edge).transpose(1,2,0)
                edge = align_to_num(edge)
                edge = to_tensor(edge, device)
                # output
                if r == 'rain':
                    output = input.clone()
                elif 'CycleGAN' in r:
                    cyclegan_output = CycleGAN(input)
                    cyclegan_output = util.tensor2im(cyclegan_output)
                    cyclegan_output = cyclegan_output[:,:,::-1]
                    cyclegan_output = align_to_num(cyclegan_output)
                    output = to_tensor(cyclegan_output, device)
                else:
                    output = unfairGAN(input, rm=rainmap, ed=edge)
                # measurement
                infer_time = (time.time() - start_time)
                psnr, ssim = batch_psnr_ssim(output.clamp(0., 1.), cal_target.clamp(0., 1.), batch_ssim=True)
                # save output
                if opt.save_img and r != 'rain':
                    os.makedirs(os.path.join('%s/%s/%s' % (opt.out,testset, r)), exist_ok=True)
                    out_path = os.path.join('%s/%s/%s/%s' % (opt.out,testset, r, img))
                    logimg = make_grid(output.data.clamp(0., 1.), nrow=8, normalize=False, scale_each=False)
                    save_image(logimg, out_path)

                results[r][testset]['psnr'].append(psnr.mean())
                results[r][testset]['ssim'].append(ssim.mean())
                results[r][testset]['time'].append(infer_time)

                print('%s,  %s,  PSNR=%.2f, SSIM=%.4f, RUNTIME=%.4f s' % (r, img, psnr, ssim, infer_time))
        # Free GPU
        if 'UnfairGAN_raindrop' in r:
            del unfairGAN
        if 'CycleGAN' in r:
            del CycleGAN
        torch.cuda.empty_cache()

# done
# np.save(os.path.join('results', 'results.npy'), results)
sys.stdout.write('\n')

for testset in testsets:
    print('########    %s    #######'%testset)
    for r in dicts:
        psnr = np.array(results[r][testset]['psnr']).mean()
        ssim = np.array(results[r][testset]['ssim']).mean()
        time = np.array(results[r][testset]['time']).mean()
        print('%20s,  %s,       PSNR:%.2f, SSIM:%.4f, RUNTIME=%.4f s' % ( r, testset, psnr, ssim, time))

