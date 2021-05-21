import cv2 as cv
import torch.nn.functional as F
import numpy as np
from skimage.io import imread, imshow
from PIL import Image

#用于损失函数的高斯金字塔
def pyramid(image):
    level = 3
    temp = image.copy()
    pyramid_images = []
    pyramid_images.append(temp)
    for i in range(level):
        dst = cv.pyrDown(temp)
        #print(dst.shape)
        pyramid_images.append(dst)
        #cv.imshow("pyramid_down_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images

#标准的高斯金字塔过程
def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        #print(dst.shape)
        pyramid_images.append(dst)
        #cv.imshow("pyramid_down_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images

#拉普拉斯金字塔
def lapalian_demo(image):
    # cv.imshow("dst", dst)
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    lapalian_images = []
    lapalian_images.append(pyramid_images[level-1])
    for i in range(level - 1, -1, -1):
        if (i - 1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            #print(i, "lapalian", lpls.shape)
            #cv.imshow("lapalian_down_" + str(i), lpls)
            #cv.waitKey()
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            #print(i, "lapalian_1", lpls.shape)
            #cv.imshow("lapaliandown" + str(i), lpls)
            #cv.waitKey()
        lapalian_images.append(lpls)
    return lapalian_images

#image = imread("images/" + "512512.jpg")
#print(image.shape)
#print(lapalian_demo(image)[3].shape)

#image = imread("data/1/" + "label.jpg")
#image = cv.resize(image, (512, 512), 0, 0, cv.INTER_AREA)  # 重新改变图像的尺寸，求拉普拉斯塔输入图像必须为方形
#print(image.shape)
#image_sequence = lapalian_demo(image)
#for i in range(4):
    #cv.imshow("lapaliandown" + str(i), image_sequence[i])
    #cv.waitKey()

#img = imread("images/" + "512512.jpg")
#img_Image = Image.fromarray(np.uint8(img))
#img2 = Image.open("images/" + "512512.jpg")

#print(img_Image)
#print(img2)