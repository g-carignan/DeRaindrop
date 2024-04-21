#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time
import os
import argparse
#Models lib
from models import *
#Metrics lib
from metrics import calc_psnr, calc_ssim

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args

def align_to_four(img):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img

def display_four_images(imgHPSNR, imgLPSNR, imgHSSIM, imgLSSIM, hpsnr, lpsnr, hssim, lssim):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(imgHPSNR)
    axes[0, 0].set_title('Highest PSNR: ' + str(hpsnr))
    axes[0, 0].axis('off')
    axes[0, 1].imshow(imgLPSNR)
    axes[0, 1].set_title('Lowest PSNR: ' + str(lpsnr))
    axes[0, 1].axis('off')
    axes[1, 0].imshow(imgHSSIM)
    axes[1, 0].set_title('Highest SSIM: ' + str(hssim))
    axes[1, 0].axis('off')
    axes[1, 1].imshow(imgLSSIM)
    axes[1, 1].set_title('Lowest SSIM: ' + str(lssim))
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def predict(image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()

    out = model(image)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :]*255.
    
    return out

def predict_with_attention(image):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()

    masks, frame1, frame2, out = model(image)  

    masks = [mask[0, 0].cpu().detach().numpy() for mask in masks]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :] * 255.

    return masks, out

def display_attention_and_result(img, attention_maps, result):
    num_maps = len(attention_maps)
    num_cols = num_maps + 1
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))

    # Input image
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Attention maps
    for i in range(num_maps):
        axes[i+1].imshow(attention_maps[i], cmap='inferno', interpolation='nearest')
        axes[i+1].set_title(f'Attention Map {i+1}')
        axes[i+1].axis('off')

    # Generated result
    axes[-1].imshow(result)
    axes[-1].set_title('Generated Result')
    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    args = get_args()
    avg_ssim_list = []
    avg_psnr_list = []
    highest_ssim = {'filename': None, 'ssim': 0}
    lowest_ssim = {'filename': None, 'ssim': np.inf}
    highest_psnr = {'filename': None, 'psnr': 0}
    lowest_psnr = {'filename': None, 'psnr': np.inf}

    model = Generator().cuda()
    model.load_state_dict(torch.load('./weights/gen.pkl'))

    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            img = align_to_four(img)
            #result = predict(img)
            #img_name = input_list[i].split('.')[0]
            #cv2.imwrite(args.output_dir + img_name + '.jpg', result)
            attention_maps, result = predict_with_attention(img)
            display_attention_and_result(img, attention_maps, result)
            img_name = input_list[i].split('.')[0]
            cv2.imwrite(args.output_dir + img_name + '.jpg', result)

    elif args.mode == 'test':
        for r in range(5):  # Run test mode 10 times
          input_list = sorted(os.listdir(args.input_dir))
          gt_list = sorted(os.listdir(args.gt_dir))
          num = len(input_list)
          cumulative_psnr = 0
          cumulative_ssim = 0
          for i in range(num):
            #print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            gt = cv2.imread(args.gt_dir + gt_list[i])
            img = align_to_four(img)
            gt = align_to_four(gt)
            result = predict(img)
            result = np.array(result, dtype='uint8')
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            #print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim

            # Update highest and lowest SSIM and PSNR images
            if cur_ssim > highest_ssim['ssim']:
              highest_ssim['filename'] = input_list[i]
              highest_ssim['ssim'] = cur_ssim
            if cur_ssim < lowest_ssim['ssim']:
              lowest_ssim['filename'] = input_list[i]
              lowest_ssim['ssim'] = cur_ssim
            if cur_psnr > highest_psnr['psnr']:
              highest_psnr['filename'] = input_list[i]
              highest_psnr['psnr'] = cur_psnr
            if cur_psnr < lowest_psnr['psnr']:
              lowest_psnr['filename'] = input_list[i]
              lowest_psnr['psnr'] = cur_psnr

          avg_psnr_list.append(cumulative_psnr / num)
          avg_ssim_list.append(cumulative_ssim / num)

          # Calculate average SSIM and PSNR
          avg_ssim = np.mean(avg_ssim_list)
          avg_psnr = np.mean(avg_psnr_list)




        # Display results
        print("Average SSIM:", avg_ssim)
        print("Average PSNR:", avg_psnr)
        highest_ssim_img = cv2.imread(args.input_dir + highest_ssim['filename'])
        lowest_ssim_img = cv2.imread(args.input_dir + lowest_ssim['filename'])
        highest_psnr_img = cv2.imread(args.input_dir + highest_psnr['filename'])
        lowest_psnr_img = cv2.imread(args.input_dir + lowest_psnr['filename'])
        display_four_images(highest_psnr_img, lowest_psnr_img, highest_ssim_img, lowest_ssim_img, highest_psnr['psnr'], lowest_psnr['psnr'], highest_ssim['ssim'], lowest_ssim['ssim'])


    else:
        print ('Mode Invalid!')
