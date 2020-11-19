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
parser.add_argument("--testset",  default='real_rain', const='all', nargs='?', choices=['real_rain'], help='')
parser.add_argument("--model",  default='all', const='all', nargs='?', choices=['all',
                                                                                'UnfairGAN',
                                                                                'AttenGAN',
                                                                                'RoboCar',
                                                                                'Pix2Pix',], help='')
parser.add_argument("--gpu", type='bool', default=True, help='')

parser.set_defaults(feature=True)

opt = parser.parse_args()
print(opt)

if opt.model == 'all':
    gen_specific = [
        'UnfairGAN',
        'AttenGAN',
        'RoboCar',
        'Pix2Pix',
    ]
else:
    gen_specific = [
        opt.model,
        ]

if opt.testset == 'all':
    testsets = [
        'rainH',
        'rainN',
        'rainL',]
else:
    testsets = [opt.testset]

## GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.gpu)else "cpu")
results = {}

sys.stdout.write('> Run ...')
with torch.no_grad():
    inxseg_chs = 0
    for r in gen_specific:
        results[r] = {}
        # estNet
        from network.unfairGan import Generator
        estNet = Generator(inRM_chs=0, inED_chs=0, mainblock_type='U_D', act_type='ReLU').to(device)
        estNet = nn.DataParallel(estNet, device_ids=[device])
        estNet.load_state_dict(torch.load('weight/U_D.pth'))
        estNet.eval()
        # rcf net
        rcfNet = RCF().to(device)
        rcfNet.load_state_dict(torch.load('weight/RCF.pth')['state_dict'])
        rcfNet.eval()
        # Gnet
        if 'AttenGAN' in r:
            from network.attentionGan.generator import Generator

            Gnet = Generator().to(device)
        elif 'RoboCar' in r:
            from network.RoboCar.generator import Derain_GlobalGenerator
            Gnet = Derain_GlobalGenerator(input_nc=3, output_nc=3, ngf=64, n_downsampling=4, n_blocks=9,
                                              norm_layer=nn.BatchNorm2d,
                                              padding_type='reflect').to(device)
        elif 'Pix2Pix' in r:
            from network.Pix2Pix.networks import define_G
            Gnet = define_G(3, 3, 128, 'batch', False, 'normal', 0.02, gpu_id=device)
        else:
            from network.unfairGan import Generator
            if r == 'U':
                Gnet = Generator(inRM_chs=0, inED_chs=0, mainblock_type='U', act_type='ReLU', ).to(device)
            if r == 'U_D':
                Gnet = Generator(inRM_chs=0, inED_chs=0, mainblock_type='U_D', act_type='ReLU', ).to(device)
            if r == 'U_D_G':
                Gnet = Generator(inRM_chs=0, inED_chs=0, mainblock_type='U_D', act_type='ReLU', ).to(device)
            if r == 'U_D_ReLU_G':
                Gnet = Generator(inRM_chs=1, inED_chs=3, mainblock_type='U_D', act_type='ReLU', ).to(device)
            if r == 'U_D_ReLU_UG':
                Gnet = Generator(inRM_chs=1, inED_chs=3, mainblock_type='U_D', act_type='ReLU', ).to(device)
            if r == 'U_D_XU_UG':
                Gnet = Generator(inRM_chs=1, inED_chs=3, mainblock_type='U_D', act_type='XU', ).to(device)
            if r == 'UnfairGAN':
                Gnet = Generator(inRM_chs=1, inED_chs=3, mainblock_type='U_D', act_type='DAF', ).to(device)

        if r != 'rain':
            Gnet = nn.DataParallel(Gnet, device_ids=[device])
            Gnet.load_state_dict(torch.load('weight/%s.pth' % r))
            Gnet.eval()

        for testset in testsets:
            ls = os.listdir('testsets/%s/rain' % testset)
            print(testset, len(ls))
            results[r][testset] = {'psnr': [], 'ssim': []}
            for i, img in enumerate(ls):
                # input
                img_rain_cv2 = cv2.imread(os.path.join('testsets/%s/rain' % testset, img))
                input = align_to_num(img_rain_cv2)
                input = to_tensor(input, device)
                # input align to 16
                input_a16 = align_to_num(img_rain_cv2,16)
                input_a16 = to_tensor(input_a16, device)

                if r in ('UnfairGAN', ):
                    # rainmap
                    est = estNet(input)
                    logimg = make_grid(est.data.clamp(0., 1.), nrow=8, normalize=False, scale_each=False)
                    est = logimg.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[:,
                          :, ::-1]
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
                    edge = np.stack([edge, edge, edge])
                    edge = np.array(edge).transpose(1, 2, 0)
                    edge = align_to_num(edge)
                    edge = to_tensor(edge, device)

                # output
                if r == 'U_D_G' or r == 'U_D' or r == 'U' or 'Pix2Pix' in r:
                    output = Gnet(input)
                elif 'RoboCar' in r:
                    output = Gnet(input_a16)

                elif 'AttenGAN' in r:
                    output = Gnet(input)[-1]
                else:
                    output = Gnet(input, rm=rainmap, ed=edge)
                os.makedirs(os.path.join('%s/%s/%s' % (opt.out,testset, r)), exist_ok=True)
                out_path = os.path.join('%s/%s/%s/%s' % (opt.out,testset, r, img))
                logimg = make_grid(output.data.clamp(0., 1.), nrow=8, normalize=False, scale_each=False)
                save_image(logimg, out_path)

                print('%s,  %s,  DONE' % (r, img))
        # Free GPU
        if r != 'rain':
            del Gnet
            torch.cuda.empty_cache()

