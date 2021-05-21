import os
import time
import torch
import cv2 as cv
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, utils
from ImageDataset import ImageSeqDataset
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
from model import PEC_model
from Loss import Pyramid_Loss, Reconstruction_Loss


#初始化参数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #载入模型
    wulalala = PEC_model().cuda()
    wulalala.apply(weights_init)
    if config.load_pretrain == True:
        wulalala.load_state_dict(torch.load(config.pretrain_dir))

    #数据集处理
    train_transform = transforms.Compose([
        BatchToTensor(),
    ])

    # 数据集路径
    #datapath = "./data/"
    # 构建数据集
    train_data = ImageSeqDataset(csv_file=os.path.join(config.datapath, 'test.txt'),
                                 img_dir=config.datapath,
                                 transform=train_transform)
    train_loader = DataLoader(train_data,
                              batch_size=config.train_batch_size,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              shuffle=True)



    #初始化损失函数
    rec_loss = Reconstruction_Loss().cuda()
    pyr_loss = Pyramid_Loss().cuda()

    optimizer = torch.optim.Adam(wulalala.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    wulalala.train()

    for epoch in range(config.num_epochs):
        for step, sample_batched in enumerate(train_loader):

            train_image, label_image = sample_batched['train'], sample_batched['label']
            for i in range(4):
                train_image[i] = train_image[i].cuda()
                label_image[i] = label_image[i].cuda()
            x1, x2, x3, x4 = wulalala(train_image[0], train_image[1], train_image[2], train_image[3])
            label_image[3] = F.interpolate(label_image[3], (2 * label_image[3].size()[2], 2 * label_image[3].size()[3]),
                                           mode='bilinear')
            label_image[2] = F.interpolate(label_image[3], (2 * label_image[2].size()[2], 2 * label_image[2].size()[3]),
                                           mode='bilinear')
            label_image[1] = F.interpolate(label_image[3], (2 * label_image[1].size()[2], 2 * label_image[1].size()[3]),
                                           mode='bilinear')
            loss = rec_loss(x4, label_image[0]) + pyr_loss(x3, x2, x1, label_image[1], label_image[2], label_image[3])


            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(wulalala.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((step + 1) % config.display_iter) == 0:
                print("Loss at iteration", step + 1, ":", loss.item())
            if ((step + 1) % config.snapshot_iter) == 0:
                torch.save(wulalala.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')



if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--datapath', type=str, default="data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.9)
	parser.add_argument('--grad_clip_norm', type=float, default=1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=2)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= True)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)




'''
high_size = 512
low_size = 128

train_transform = transforms.Compose([
    BatchToTensor(),
])

label_transform = transforms.Compose([
    BatchToTensor(),
])

train_batch_size = 1
test_batch_size = 1

#数据集路径
datapath = "./data/"
#构建数据集
train_data = ImageSeqDataset(csv_file=os.path.join(datapath, 'test.txt'),
                                          img_dir = datapath,
                                          transform = train_transform)
train_loader = DataLoader(train_data,
                               batch_size=train_batch_size,
                               shuffle=False,
                               pin_memory=True,
                               num_workers=1)


def main(argv=None):
    for step, sample_batched in enumerate(train_loader, 0):
        train_image, label_image = sample_batched['train'], sample_batched['label']
        wulalala = PEC_model()
        rec_loss = Reconstruction_Loss()
        pyr_loss = Pyramid_Loss()
        x1,x2,x3,x4 = wulalala(train_image[0], train_image[1], train_image[2], train_image[3])
        print(step, x1.size())
        print(step, x2.size())
        print(step, x3.size())
        print(step, x4.size())
        print("ok")
        label_image[3] = F.interpolate(label_image[3], (2*label_image[3].size()[2], 2*label_image[3].size()[3]), mode='bilinear')
        label_image[2] = F.interpolate(label_image[3], (2*label_image[2].size()[2], 2*label_image[2].size()[3]), mode='bilinear')
        label_image[1] = F.interpolate(label_image[3], (2*label_image[1].size()[2], 2*label_image[1].size()[3]), mode='bilinear')
        print(label_image[3].size())
        print(label_image[2].size())
        print(label_image[1].size())
        print(label_image[0].size())

        loss = rec_loss(x4, label_image[0]) + pyr_loss(x3, x2, x1, label_image[1], label_image[2], label_image[3])
        print("loss", loss)

if __name__=='__main__':
    main()
'''

