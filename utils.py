import cv2
import numpy as np
import torch
from skimage.measure import compare_psnr, compare_ssim


def batch_psnr_ssim(img, imclean, data_range=1., batch_ssim=True):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the x image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.cpu().data.numpy().astype(np.float32)
    imgclean = imclean.cpu().data.numpy().astype(np.float32)
    psnrs = []
    ssims = []
    for i in range(img_cpu.shape[0]):
        # psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
        im1 = np.array(img_cpu[i, :, :, :].transpose((1, 2, 0)) * 255., dtype='uint8')
        im2 = np.array(imgclean[i, :, :, :].transpose((1, 2, 0)) * 255., dtype='uint8')
        im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        psnrs.append(compare_psnr(im1_y, im2_y))
        if batch_ssim:
            ssims.append(compare_ssim(im1_y, im2_y))

    return np.array(psnrs), np.array(ssims)


def align_to_num(img, num=4):
    a_row = int(img.shape[0] / num) * num
    a_col = int(img.shape[1] / num) * num
    img = img[0:a_row, 0:a_col]
    return img

def to_tensor(img, gpu):
    img = np.array(img[:, :, ::-1] / 255.0).astype('float32')
    img = img.transpose(2, 0, 1)
    img = torch.Tensor(img).unsqueeze(0)
    img = img.to(gpu)
    return img

def make_rainmap(img_rain,img_drain):
    temp = np.abs(align_to_num(img_rain).copy() - img_drain.copy())
    cond = (temp > 20) * (temp < 230)
    cond = cond[:, :, 0] * cond[:, :, 1] * cond[:, :, 2]
    rm1 = np.zeros_like(align_to_num(img_rain))[:, :, 0]
    rm1[cond] = 30
    rainmap = np.expand_dims(rm1.copy(), axis=-1)
    return  rainmap


def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im