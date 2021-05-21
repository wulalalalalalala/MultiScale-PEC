import os
import torch
import pandas as pd
import cv2 as cv
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread, imshow
from LaplacianPyramid import pyramid_demo, lapalian_demo, pyramid

class ImageSeqDataset(Dataset):
	def __init__(self, csv_file,
				 img_dir,
				 transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			hr_img_seq_dir (string): Directory with all the high resolution image sequences.
			transform (callable, optional): transform to be applied on a sample.
		"""
		self.seqs = pd.read_csv(csv_file, sep='\n', header=None)
		self.image_root = img_dir
		self.transform = transform

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			samples: a Tensor that represents a video segment.
		"""
		hr_seq_dir = os.path.join(self.image_root, str(self.seqs.iloc[index, 0]))
		train_image = imread(os.path.join(hr_seq_dir, 'train.jpg'))
		#print(type(train_image ))
		train_image = cv.resize(train_image, (512, 512), 0, 0, cv.INTER_AREA)  # 重新改变图像的尺寸，求拉普拉斯塔输入图像必须为方形
		#print(type(train_image))
		label_image = imread(os.path.join(hr_seq_dir, 'label.jpg'))
		label_image = cv.resize(label_image, (512, 512), 0, 0, cv.INTER_AREA)  # 重新改变图像的尺寸，求拉普拉斯塔输入图像必须为方形
		train_seq_image = lapalian_demo(train_image)
		label_seq_image = pyramid(label_image)
		#print(len(train_seq_image))
		#print(type(train_seq_image[0]))
		#print(len(label_seq_image))
		#print(type(label_seq_image[0]))
		#a = transforms.ToTensor()(train_seq_image[0])
		#print(a.shape)

		if self.transform is not None:
			train_seq_image  = self.transform(train_seq_image )
			label_seq_image = self.transform(label_seq_image)
		#print(label_seq_image[0].shape)
		sample = {'train': train_seq_image, 'label': label_seq_image}
		return sample

	def __len__(self):
		return len(self.seqs)

	@staticmethod
	def _reorderBylum(seq):
		I = torch.sum(torch.sum(torch.sum(seq, 1), 1), 1)
		_, index = torch.sort(I)
		result = seq[index, :]
		return result
