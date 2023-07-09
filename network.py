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


def send_pid_to_measure(PID):
    IP = '127.0.0.1'
    PORT = 5005
    MESSAGE = f'{PID}'
    MESSAGE = MESSAGE.encode()
    #sending the pid to measuring app
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((IP, PORT))
            s.sendall(MESSAGE)
        except ConnectionRefusedError:
            print("Could not connect to measure process. Continuing.")


def run(argv):

    PID = os.getpid()

    paths = glob.glob("unlabeled2017/*.jpg")

    num_of_images = 90_000#int(argv[1])

    np.random.seed(42)
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
            if split == 'train':
                self.transforms = transforms.Compose([
                    transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                ])
            elif split == 'val':
                self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
            
            self.split = split
            self.size = SIZE
            self.paths = paths
        
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transforms(img)
            img = np.array(img)
            img_lab = rgb2lab(img).astype("float32")
            img_lab = transforms.ToTensor()(img_lab)
            L = img_lab[[0], ...] / 50. - 1. 
            ab = img_lab[[1, 2], ...] / 128.
            return {'L': L, 'ab': ab}
        
        def __len__(self):
            return len(self.paths)

    batch_size = 32
    val_batch_size = 8

    dataset = ColorizationDataset(train_paths, split="train")
    dataloader_train = DataLoader(dataset, batch_size=batch_size, num_workers=16,
                                pin_memory=True)


    dataset = ColorizationDataset(val_paths, split="val")
    dataloader_val = DataLoader(dataset, batch_size=val_batch_size, num_workers=16,
                                pin_memory=True)



    with torch.no_grad():
        torch.cuda.empty_cache()

    start_time = str(datetime.now())[:-7]

    device = torch.device("cuda:0")

    #print(argv[2] == "True")
    model = UNet(smaller=False)
    #model = UNet(smaller=argv[2] == "True")
    model.to(device)
    optim = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    #optim = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0001, fused=True)
    optim.zero_grad()

    ssim = StructuralSimilarityIndexMeasure().to(device)

    psnr = PeakSignalNoiseRatio().to(device)

    send_pid_to_measure(PID)

    for epoch in range(5):

        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):
            model.train()
            black_white = data['L']
            black_white = black_white.to(device)
            true_pic = data['ab']
            true_pic = true_pic.to(device)


            outputs = model(black_white)
            loss = torch.nn.MSELoss()
            loss_output = loss(outputs, true_pic)
            loss_output.backward()
            optim.step()
            optim.zero_grad()
            

            running_loss += loss_output.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        model.eval()
        val_loss = 0.0

        for j, data in enumerate(dataloader_val, 0):
            with torch.no_grad():
                torch.cuda.empty_cache()
            black_white = data['L']
            black_white_gpu = black_white.to(device)
            true_pic = data['ab']
            true_pic_gpu = true_pic.to(device)

            outputs = model(black_white_gpu)
            loss = torch.nn.MSELoss()
            loss_output = loss(outputs, true_pic_gpu)
            val_loss += loss_output.item()
            ssim.update(outputs, true_pic_gpu)
            psnr.update(outputs, true_pic_gpu)
        print(f'[{epoch + 1}] validation loss: {val_loss / ((num_of_images-num_train)/val_batch_size):.5f} validation ssim: {ssim.compute():.5f} validation psnr: {psnr.compute():.5f}')
    end_time = str(datetime.now())[:-7]
    print(start_time)
    print(end_time)
    #torch.save(model.state_dict(), "output/model")


if __name__ == "__main__":
    run()