import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import cv2 as cv
from skimage.io import imread, imshow, imsave
import sys
import argparse
import time
from model import PEC_model
from LaplacianPyramid import lapalian_demo
import numpy as np
from torchvision import transforms
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
from PIL import Image
import glob
import time

test_transform = transforms.Compose([
    BatchToTensor(),
])


# snapshots-primary/Epoch199.pth
# snapshots-without-exp/Epoch199.pth
# snapshots-without-spa/Epoch199.pth

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = imread(image_path)

    image_sequence = lapalian_demo(data_lowlight)
    #cv.imshow("lapaliandown" + str(0), image_sequence[0])
    #cv.waitKey()
    image_sequence = test_transform(image_sequence)
    for i in range(4):
        image_sequence[i] = image_sequence[i].cuda().unsqueeze(0)

    testmodel = PEC_model().cuda()
    testmodel.load_state_dict(torch.load('snapshots/Epoch124.pth'))
    start = time.time()
    out1, out2, out3, enhanced_image = testmodel(image_sequence[0], image_sequence[1], image_sequence[2], image_sequence[3])

    end_time = (time.time() - start)
    print(end_time)
    result_path = image_path.replace('test_data', 'results')
    result_path = ".\\" + result_path
    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = 'test_data/'
        test_list = glob.glob(filePath + "/*")
        for image in test_list:
            # image = image
            print(image)
            lowlight(image)