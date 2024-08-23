import argparse
import os
import numpy as np
from tqdm import tqdm  # tqdm 라이브러리 추가

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# 디렉토리 경로 변수로 지정
image_dir = 'images/upsamplinglayer'
model_dir = 'saved_models/upsamplinglayer'
lr_dir = '/home/wj/works/SR-project/WSdata/LR'
hr_dir = '/home/wj/works/SR-project/WSdata/HR'

# 디렉토리 생성
os.makedirs(image_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0002, help='adma: learning rate')
parser.add_argument('--hr_height', type=int, default=100, help='high res. image height')
parser.add_argument('--hr_width', type=int, default=100, help='high res. image width')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between saving image samples')
parser.add_argument('--residual_blocks', type=int, default=16, help='number of residual blocks in the generator')
parser.add_argument('--warmup_batches', type=int, default=2000, help='number of batches with pixel-wise loss only')
parser.add_argument('--lambda_adv', type=float, default=0.01, help='adversial loss weight')
parser.add_argument('--lambda_pixel', type=float, default=1, help='pixel-wise loss weight')
parser.add_argument('--lambda_content', type=float, default=1, help='content loss weight')
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hr_shape = (opt.hr_height, opt.hr_width)
channels = 3

# initialize generator and discriminator  
generator = GeneratorRRDB(channels, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# set feature extractor to inference mode
feature_extractor.eval()

# Losses  
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device) 
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

# 모델이 이미 존재하는 경우, 해당 모델 불러오기
if opt.epoch != 0:
    generator.load_state_dict(torch.load(f'{model_dir}/generator_{opt.epoch}.pth'))
    # discriminator.load_state_dict(torch.load(f'{model_dir}/discriminator_{opt.epoch}.pth'))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# 데이터셋 로드
train_dataset = TrainDatasetFromFolder(lr_dir, hr_dir)
train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=32, shuffle=True)

# ----------
# Training  
# ----------  

for epoch in range(opt.epoch, opt.n_epochs):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{opt.n_epochs}")  # tqdm으로 진행률 바 생성
    for i, (data, target) in progress_bar:

        batches_done = epoch * len(train_loader) + i

        imgs_lr = Variable(data.type(Tensor))
        imgs_hr = Variable(target.type(Tensor))

        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------   

        optimizer_G.zero_grad()

        gen_hr = generator(imgs_lr)

        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            loss_pixel.backward()
            optimizer_G.step()
            progress_bar.set_postfix({'G pixel': loss_pixel.item()})  # 진행률 바에 손실 정보 표시
            continue

        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        loss_GAN = (criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) + 
                    criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)) / 2

        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        loss_G = opt.lambda_content * loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # --------------------
        # Train Discriminator  
        # --------------------  

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # 진행률 바에 손실 정보 업데이트
        progress_bar.set_postfix({'D loss': loss_D.item(), 'G loss': loss_G.item()})

        if batches_done % opt.sample_interval == 0:
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=5, mode='bicubic')
            img_grid = torch.clamp(torch.cat((imgs_lr, gen_hr, imgs_hr), -1), min=0, max=1)
            save_image(img_grid, f'{image_dir}/{batches_done}.png', nrow=1, normalize=False)

    torch.save(generator.state_dict(), f'{model_dir}/generator_{epoch}.pth')
    # torch.save(discriminator.state_dict(), f'{model_dir}/discriminator_{epoch}.pth')
