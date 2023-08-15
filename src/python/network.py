import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from unet import *
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import socket
from datetime import datetime
import cv2

def train_one_epoch(model, device, optim, dataloader_train, autocast, scaler, batches_threshold, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader_train, 0):
        optim.zero_grad()

        black_white = data['L']
        black_white = black_white.to(device)
        true_pic = data['ab']
        true_pic = true_pic.to(device)

        if autocast:
            with torch.autocast('cuda'):
                outputs = model(black_white)
                loss = torch.nn.MSELoss()
                loss_output = loss(outputs, true_pic)
            scaler.scale(loss_output).backward()
            scaler.step(optim)
            scaler.update()

        else:
            outputs = model(black_white)
            loss = torch.nn.MSELoss()
            loss_output = loss(outputs, true_pic)
        
            loss_output.backward()
            optim.step()
        

        running_loss += loss_output.item()

        if i % batches_threshold == batches_threshold - 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batches_threshold :.3f}')
            running_loss = 0.0
            
def check_validation(model, device, dataloader_val, autocast, scaler, ssim, psnr, epoch, num_of_images, num_train):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for j, data in enumerate(dataloader_val, 0):
            black_white = data['L']
            black_white = black_white.to(device)
            true_pic = data['ab']
            true_pic = true_pic.to(device)
            
            if autocast:
                with torch.autocast('cuda'):
                    outputs = model(black_white)
                    loss = torch.nn.MSELoss()
                    loss_output = loss(outputs, true_pic)
            else:
                outputs = model(black_white)
                loss = torch.nn.MSELoss()
                loss_output = loss(outputs, true_pic) 
            
            val_loss += loss_output.item()
            ssim.update(outputs, true_pic)
            psnr.update(outputs, true_pic)
        val_loss = val_loss / ((num_of_images-num_train)/128)
        val_ssim = ssim.compute()
        val_psnr = psnr.compute()
        print(f'[{epoch + 1}] validation loss: {val_loss:.5f} validation ssim: {val_ssim:.5f} validation psnr: {val_psnr:.5f}')
    ssim.reset()
    psnr.reset()
    return val_loss, val_ssim, val_psnr

def run(argv):
    num_of_images = argv[0]
    smaller = argv[1]
    epochs = argv[2]
    autocast = argv[3]
    batch = argv[4]
    adam = argv[5]
    print(argv)

    paths = glob.glob("unlabeled2017/*.jpg")

    paths_subset = np.random.choice(paths, num_of_images, replace=False)

    num_train = int(num_of_images*0.8)

    rand_idxs = np.random.permutation(num_of_images)
    train_idxs = rand_idxs[:num_train]
    val_idxs = rand_idxs[num_train:]
    train_paths = paths_subset[train_idxs] 
    val_paths = paths_subset[val_idxs]
    print(len(train_paths), len(val_paths))

    SIZE = 128

    class ColorizationDataset(Dataset):
        def __init__(self, paths, split='train'):
            self.split = split
            self.SIZE = SIZE
            self.paths = paths
        
        def __getitem__(self, idx):
            img = cv2.imread(self.paths[idx])
            img =  cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            img = (img/128.)-1
            img = transforms.ToTensor()(img)
            L = img[[0], ...]
            ab = img[[1, 2], ...]
            return {'L': L, 'ab': ab}
        
        def __len__(self):
            return len(self.paths)

    batch_size = batch
    val_batch_size = batch


    dataset = ColorizationDataset(train_paths, split="train")
    dataloader_train = DataLoader(dataset, batch_size=batch_size, num_workers=16,
                                pin_memory=True)


    dataset = ColorizationDataset(val_paths, split="val")
    dataloader_val = DataLoader(dataset, batch_size=128, num_workers=16,
                                pin_memory=True)

    batches_threshold = 6_400//batch_size


    with torch.no_grad():
        torch.cuda.empty_cache()

    start_time = str(datetime.now())[:-7]

    device = torch.device("cuda:0")

    model = UNet(smaller=smaller)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0) if adam else torch.optim.RMSprop(model.parameters(), lr=0.0001)
    optim.zero_grad()

    ssim = StructuralSimilarityIndexMeasure().to(device)

    psnr = PeakSignalNoiseRatio().to(device)

    scaler = torch.cuda.amp.GradScaler() if autocast else None

    for epoch in range(epochs):
        train_one_epoch(model, device, optim, dataloader_train, autocast, scaler, batches_threshold, epoch)

        val_loss, val_ssim, val_psnr = check_validation(model, device, dataloader_val, autocast, scaler, ssim, psnr, epoch, num_of_images, num_train)

    end_time = str(datetime.now())[:-7]
    print(start_time)
    print(end_time)
    torch.save(model.state_dict(), "output/model")
    return start_time, end_time, val_loss, val_ssim, val_psnr
    


if __name__ == "__main__":
    run((30000, False))