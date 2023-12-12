import os
import math
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

# 读取文件
kaggle_3m = './kaggle_3m/'

# 读取kaggle_3m中的每个子文件（‘*'是为了每个都读取）
dirs = glob.glob(kaggle_3m+'*')

# 用来测试读取成功没
# os.listdir('.kaggle_3m\\TCGA_CS_4941_19960909')

data_img = []  # 存放原始图像
data_label = []  # 存放筛选的图像（即含有mask)
for subdir in dirs:
    dirname = subdir.split('\\')[-1]
    # print(dirname)
    for filename in os.listdir(subdir):
        img_path = subdir+'/'+filename
        # print(img_path)
        if 'mask' in img_path:
            data_label.append(img_path)
        else:
            data_img.append(img_path)

# 用来测试读取成功没：
# print(data_img),print(data_label)
# 查看数据数量是否对等：(目前是对等的，都是3929）
# print(len(data_label)),print(len(data_img))

# 如果不对等，需要做更改  (并且构建一一对应的关系！！）
data_imgX = []
for i in range(len(data_label)):
    img_mask = data_label[i]
    img = img_mask[:-9]+'.tif'
    data_imgX.append(img)
    # print(data_imgX)
# print(data_imgX[10:15])
# print(data_label[10:15])

data_newimg = []
data_newlabel = []
# 读取不全为0的像素图片，防止对训练的影响
for i in data_label:
    value = np.max(cv2.imread(i))
    # print(value)
    try:
        if value>0:
            data_newlabel.append(i)
            i_img = i[:-9]+'.tif'
            data_newimg.append(i_img)
    except:
        pass
# 验证长度，都是1373
# print(len(data_newimg))
# print(len(data_newlabel))

# print(data_newimg)
# print(data_newlabel)
# im = Image.open('./kaggle_3m\\TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif')

# 观察是否正确读取像素
# img = cv2.imread('./kaggle_3m\\TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif')
# print(img)

train_transformer = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
test_transformer = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
class BrainMRIdataset(Dataset):
    def __init__(self,img,mask,transformer):
        self.img = img
        self.mask = mask
        self.transformer = transformer

    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]

        img_open = Image.open(img)
        img_tensor = self.transformer(img_open)

        mask_open = Image.open(mask)
        mask_tensor = self.transformer(mask_open)

        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)  # 即把通道数去掉

        return img_tensor,mask_tensor

    def __len__(self):
        return len(self.img)

# 例如：[0,1,2,3,4,5,6]转化为下面二维（二分类）
# [[255,0,255,0]
#  [255,0,0,0]
#  [0,0,0,0]]

# 选取前1000张作为训练，后373作为测试
s=1000
train_img = data_newimg[:s]
train_label = data_newlabel[:s]
test_img = data_newimg[s:]
test_label = data_newlabel[s:]

train_data = BrainMRIdataset(train_img,train_label,train_transformer)
test_data = BrainMRIdataset(test_img,test_label,test_transformer)

dl_train = DataLoader(train_data,batch_size=8,shuffle=True)
dl_test = DataLoader(test_data,batch_size=8,shuffle=True)

img,label = next(iter(dl_train))
plt.figure(figsize=(12,8))
for i,(img,label) in enumerate(zip(img[:4],label[:4])):#4是8/2（batch_size=8)
    img = img.permute(1, 2, 0).numpy()
    # img = cv2.resize(img, (256, 256))
    label = label.numpy()
    plt.subplot(2,4,i+1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(2,4,i+5)
    plt.imshow(label)
    plt.title('Mask Image')
    plt.show()

# # 下采样，两层卷积
class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Encoder,self).__init__()
        # 定义两层卷积
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),  #填充一圈像素
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x,if_pool=True):
        # 判断是否需要maxlpool
        if if_pool:
            x = self.maxpool(x)
        x = self.conv_relu(x)
        return x
    # pool--conv==conv

# 上采样
class Decoder(nn.Module):
    def __init__(self,channels):
        super(Decoder,self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(2*channels,channels, kernel_size=3, padding=1),  # 填充一圈像素
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(channels,channels//2,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv_relu(x)
        x = self.upconv_relu(x)
        return x
# conv--conv--trans

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.encode1 = Encoder(3,64)
        self.encode2 = Encoder(64, 128)
        self.encode3 = Encoder(128, 256)
        self.encode4 = Encoder(256, 512)
        self.encode5 = Encoder(512, 1024)
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.decode1 = Decoder(512)
        self.decode2 = Decoder(256)
        self.decode3 = Decoder(128)

        self.convDouble = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Conv2d(64,32,kernel_size=1)

    def forward(self,x):
        x1 = self.encode1(x,if_pool=False)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)

        x5 = self.upconv_relu(x5)

        x5 = torch.cat([x4, x5],dim=1)
        x5 = self.decode1(x5)
        x5 = torch.cat([x3, x5], dim=1)
        x5 = self.decode2(x5)
        x5 = torch.cat([x2, x5], dim=1)
        x5 = self.decode3(x5)
        x5 = torch.cat([x1, x5], dim=1)

        x5 = self.convDouble(x5)
        x5 = self.last(x5)
        return x5

model = Unet()  # Define the model

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 10  # Define the number of epochs
#
# for epoch in range(num_epochs):
#     for batch_idx, (img, label) in enumerate(dl_train):
#         optimizer.zero_grad()
#         pred = model(img)
#         loss = criterion(pred, label)
#         loss.backward()
#         optimizer.step()
#
# # After the training loop, you can evaluate the accuracy on the test set
# with torch.no_grad():
#     total = 0
#     correct = 0
#     for img, label in dl_test:
#         pred = model(img)
#         _, predicted = torch.max(pred.data, 1)
#         total += label.size(0)
#         correct += (predicted == label).sum().item()
#     accuracy = correct / total
#     print(f"Accuracy on test set: {accuracy}")
#
# img,label = next(iter(dl_train))
# pred = model(img)
# print(pred.shape)


# state_dict = torch

