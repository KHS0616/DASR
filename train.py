"""
DAST 학습 코드

Writer : KHS0616
Last Update : 2021-10-21
"""

from ctypes import resize
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from datasets.DASR_datasets import MultiScaleSRDataset
from datasets.DASR_degrade import Degradeprocessing
from losses.DASR_loss import Loss_SR
from models.DASR_encoder import DegradationEncoder
from models.DASR_generator import DASR

import os
from tqdm import tqdm
import imageio
import cv2

class Trainer():
    def __init__(self):
        # 학습 타이틀 설정 및 작업공간 생성
        self.title = "DASR-After2"
        self.setWorkSpace()

        # 옵션 값들
        self.scale = 2

        # 학습을 위한 데이터 설정
        self.setDevice()
        self.setDataLoader()
        self.setModel()
        self.setLoss()
        self.setOptimizer()

        # LR 생성을 위한 Degrade 모듈 설정
        self.degrade = Degradeprocessing(device=self.device)

        # 테스트를 위한 이미지 생성
        image = imageio.imread("LR.png")        
        self.test_image = cv2.resize(image, (512, 512))
        self.test_image_HR = imageio.imread("HR.png")

    def setWorkSpace(self):
        """ 작업공간 설정"""
        self.workspace_path_model_encoder = os.path.join("./results", self.title, "models", "encoder")
        os.makedirs(self.workspace_path_model_encoder, exist_ok=True)

        self.workspace_path_model_whole = os.path.join("./results", self.title, "models", "whole")
        os.makedirs(self.workspace_path_model_whole, exist_ok=True)

        self.workspace_path_images = os.path.join("./results", self.title, "images")
        os.makedirs(self.workspace_path_images, exist_ok=True)

        self.workspace_path_optimizer_encoder = os.path.join("./results", self.title, "optimizer", "encoder")
        os.makedirs(self.workspace_path_optimizer_encoder, exist_ok=True)

        self.workspace_path_optimizer_whole = os.path.join("./results", self.title, "optimizer", "whole")
        os.makedirs(self.workspace_path_optimizer_whole, exist_ok=True)

    def setDataLoader(self):
        """ 데이터 로더 설정 """
        self.dataset = MultiScaleSRDataset()
        self.dataloader = DataLoader(            
            dataset=self.dataset,
            batch_size=32,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            drop_last=True
        )

    def setModel(self):
        """ Model 설정 """
        self.model_E = DegradationEncoder().to(self.device)
        self.model = DASR().to(self.device)
        self.model.load_state_dict(torch.load("./results/DASR-After/models/whole/DASR_590.pth"))
        self.model_E.load_state_dict(self.model.E.state_dict())

    def setDevice(self):
        """ Device 설정 """
        self.device = torch.device("cuda:0")

    def setLoss(self):
        """ Loss 설정 """
        self.contrast_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss_SR = Loss_SR()

    def setOptimizer(self):
        """ Optimizer 설정 """        
        self.optimizer_E = Adam(filter(lambda x: x.requires_grad, self.model_E.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler_E = StepLR(self.optimizer_E, step_size=60, gamma=0.1)

        self.optimizer = Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = StepLR(self.optimizer, step_size=125, gamma=0.5)

    def quantize(self, img, rgb_range):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    def train(self):
        """ 학습 메소드 """
        self.model.train()
        self.model_E.train()

        for epoch in range(1, 600+1, 1):
            # Scheduler Step
            if epoch <= 100:
                self.scheduler_E.step()
            elif epoch > 100:
                self.scheduler.step()
            
            with tqdm(total=len(self.dataset)//32, ncols=160) as t:
                t.set_description(f"Epoch : {epoch}/600")
                for idx, HR in enumerate(self.dataloader):
                    # HR 데이터 Device 설정
                    HR = HR.to(self.device)

                    LR, _ = self.degrade(HR)

                    # Degradation Encoder 학습
                    if epoch <= 100:
                        # Inference
                        _, output, target = self.model_E(im_q=LR[:,0,...], im_k=LR[:,1,...])

                        # Loss 측정
                        loss_contrast = self.contrast_loss(output, target)
                        loss = loss_contrast

                        # Backward
                        self.optimizer_E.zero_grad()
                        loss.backward()
                        self.optimizer_E.step()
                    
                    # Whole Network 학습
                    elif epoch > 100:
                        if epoch == 101:
                            self.model.E.load_state_dict(self.model_E.state_dict())

                        # Inference
                        SR, output, target = self.model(LR)

                        # Loss 측정
                        loss_SR = self.loss_SR(SR, HR[:,0,...])
                        loss_contrast = self.contrast_loss(output, target)
                        loss = loss_contrast + loss_SR

                        # Backward
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    t.update(1)
                    t.set_postfix({"Loss : ":"{:.6f}".format(loss)})

            # 모델 저장
            if epoch <= 100 and epoch % 10 == 0:
                torch.save(self.model_E.state_dict(), os.path.join(self.workspace_path_model_encoder, f"DASR_Encoder_{epoch}.pth"))
            elif epoch > 100 and epoch % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.workspace_path_model_whole, f"DASR_{epoch}.pth"))

            # 테스트 이미지 저장
            if epoch > 100 and epoch % 10 == 0:
                with torch.no_grad():
                    resized_lr = F.interpolate(LR[:,0,...], scale_factor=2)
                    grid_tensor = make_grid(tensor=[self.quantize(resized_lr, 255)[0], self.quantize(SR, 255)[0], self.quantize(HR[:,0,...], 255)[0]], normalize=True)
                    save_image(tensor=grid_tensor, fp=os.path.join(self.workspace_path_images, f"{epoch}.png"))

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()