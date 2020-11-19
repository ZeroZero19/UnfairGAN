import argparse
import os
import sys
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from utils import *
from network.RCF.models import RCF


def strToBool(str):
    return str.lower() in ('true', 'yes', 'on', 't', '1')
parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)
parser.add_argument("--out", type=str, default='results', help='path of save results')
parser.add_argument("--testset",  default='raindrop', const='raindrop', nargs='?', choices=['raindrop'], help='')
parser.add_argument("--model",  default='all', const='all', nargs='?', choices=['all',
                                                                                'U_D_raindrop',
                                                                                'UnfairGAN_raindrop',], help='')
parser.add_argument("--save_img", type='bool', default=True, help='')
parser.add_argument("--debug", type='bool', default=False, help='')
parser.add_argument("--gpu", type='bool', default=True, help='')
parser.set_defaults(feature=True)

opt = parser.parse_args()
print(opt)

if opt.model == 'all':
    gen_specific = [
        'rain',
        'UnfairGAN_raindrop',
    ]
else:
    gen_specific = [
        'rain',
        opt.model,
        ]
testsets = [opt.testset]

## GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.gpu)else "cpu")

results = {}

sys.stdout.write('> Run ...')
with torch.no_grad():
    for r in gen_specific:
        results[r] = {}
        from network.unfairGan import Generator
        # estNet
        estNet = Generator(inRM_chs=0, inED_chs=0, mainblock_type='U_D', act_type='ReLU', nfeats=16).to(device)
        estNet = nn.DataParallel(estNet, device_ids=[device])
        estNet.load_state_dict(torch.load('weight/U_D_raindrop.pth'))
        estNet.eval()
        # rcf net
        rcfNet = RCF().to(device)
        rcfNet.load_state_dict(torch.load('weight/RCF.pth')['state_dict'])
        rcfNet.eval()
        # unfairGAN net
        if r != 'rain':
            unfairGAN = Generator(inRM_chs=1, inED_chs=3, mainblock_type='U_D', act_type='DAF', nfeats=16).to(device)
            unfairGAN = nn.DataParallel(unfairGAN, device_ids=[device])
            unfairGAN.load_state_dict(torch.load('weight/%s.pth' % r))
            unfairGAN.eval()

        for testset in testsets:
            ls = os.listdir('testsets/%s/rain' % testset)
            print(testset, len(ls))
            results[r][testset] = {'psnr': [], 'ssim': []}
            for i, img in enumerate(ls):
                if opt.debug and i > 0: continue
                # input
                img_rain_cv2 = cv2.imread(os.path.join('testsets/%s/rain' % testset, img))
                input = align_to_num(img_rain_cv2)
                input = to_tensor(input, device)
                # input align to 16
                input_a16 = align_to_num(img_rain_cv2,16)
                input_a16 = to_tensor(input_a16, device)
                # target
                target_rgb = cv2.imread(os.path.join('testsets/%s/gt' % testset, img.replace('rain','clean')))
                target = align_to_num(target_rgb)
                target = to_tensor(target, device)
                # target align to 16
                target_a16 = align_to_num(target_rgb,16)
                target_a16 = to_tensor(target_a16, device)
                # rainmap
                est = estNet(input)
                logimg = make_grid(est.data.clamp(0., 1.), nrow=8, normalize=False, scale_each=False)
                est = logimg.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[:, :, ::-1]
                derain = align_to_num(est)
                rainmap = make_rainmap(img_rain_cv2, derain)
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
                    psnr, ssim = batch_psnr_ssim(input.clamp(0., 1.), target.clamp(0., 1.), 1., batch_ssim=True)
                else:
                    output = unfairGAN(input, rm=rainmap, ed=edge)
                    psnr, ssim = batch_psnr_ssim(output.clamp(0., 1.), target.clamp(0., 1.), 1., batch_ssim=True)
                if opt.save_img and r != 'rain':
                    os.makedirs(os.path.join('%s/%s/%s' % (opt.out,testset, r)), exist_ok=True)
                    out_path = os.path.join('%s/%s/%s/%s' % (opt.out,testset, r, img))
                    logimg = make_grid(output.data.clamp(0., 1.), nrow=8, normalize=False, scale_each=False)
                    save_image(logimg, out_path)

                results[r][testset]['psnr'].append(psnr.mean())
                results[r][testset]['ssim'].append(ssim.mean())

                print('%s,  %s,  PSNR=%.2f, SSIM=%.4f' % (r, img, psnr, ssim))
        # Free GPU
        if r != 'rain':
            del unfairGAN
            torch.cuda.empty_cache()

# done
# np.save(os.path.join('results', 'results.npy'), results)
sys.stdout.write('\n')

for testset in testsets:
    print('########    %s    #######'%testset)
    for r in gen_specific:
        psnr = np.array(results[r][testset]['psnr']).mean()
        ssim = np.array(results[r][testset]['ssim']).mean()
        print('%16s,  %s,       PSNR:%.2f, SSIM:%.4f' % ( r, testset, psnr, ssim))

