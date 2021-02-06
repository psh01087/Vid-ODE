import argparse
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from math import log10
import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms

import eval_models as models


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_image_dir', default='./')

    opt = parser.parse_args()
    return opt

def Evaluation(opt, pred_list, gt_list):
    pred_list.sort()
    gt_list.sort()

    T1 = Transforms.ToTensor()
    T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                             Transforms.ToTensor(),
                             Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])

    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)
    model.eval()

    avg_ssim, avg_mse, avg_distance = 0.0, 0.0, 0.0

    with torch.no_grad():
        print("Calculate SSIM, MSE, LPIPS...")
        for i, (pred_img, gt_img) in enumerate(zip(pred_list, gt_list)):
            # Calculate SSIM
            gt_img = Image.open(os.path.join(opt.result_image_dir, 'gt', gt_img))
            gt_np = np.asarray(gt_img.convert('L'))
            pred_img = Image.open(os.path.join(opt.result_image_dir, 'pred', pred_img))
            pred_np = np.asarray(pred_img.convert('L'))
            avg_ssim += ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)

            # Calculate LPIPS
            gt_img_LPIPS = T2(gt_img).unsqueeze(0).cuda()
            pred_img_LPIPS = T2(pred_img).unsqueeze(0).cuda()
            avg_distance += model.forward(gt_img_LPIPS, pred_img_LPIPS)

            # Calculate MSE
            gt_img_MSE = T1(gt_img).unsqueeze(0).cuda()
            pred_img_MSE = T1(pred_img).unsqueeze(0).cuda()
            avg_mse += F.mse_loss(gt_img_MSE, pred_img_MSE)

            if (i + 1) % 10 == 0:
                print("step: %8d evaluation..." % (i+1))

        avg_ssim /= len(gt_list)
        avg_mse = avg_mse / len(gt_list)
        avg_psnr = 10 * log10(1 / avg_mse)
        avg_distance = avg_distance / len(gt_list)

    print("SSIM : %f / MSE : %f / LPIPS : %f / PSNR : %f" % (avg_ssim, avg_mse, avg_distance, avg_psnr))

    return avg_ssim, avg_mse, avg_distance


def main():
    opt = get_opt()

    # Output과 Ground Truth Data
    pred_list = os.listdir(os.path.join(opt.result_image_dir, 'pred'))
    gt_list = os.listdir(os.path.join(opt.result_image_dir, 'gt'))

    avg_ssim, avg_mse, avg_distance = Evaluation(opt, pred_list, gt_list)

    print("Finish evaluate.py...")


if __name__ == '__main__':
    main()