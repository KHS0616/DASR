"""
DASR Loss 코드

Writer : KHS0616
Last Update : 2021-10-21
"""
import torch
import torch.nn as nn

class Loss_SR(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss_SR, self).__init__()
        # Loss 리스트 선언
        self.loss = []

        # 사용할 Loss 추가
        self.loss.append({
            "type" : "L1",
            "weight" : 1,
            "function" : nn.L1Loss()
        })


    def forward(self, sr, hr):
        """ Loss 순전파 """
        # Loss 리스트 선언
        losses = []

        # 저장된 Loss를 측정
        for idx, lf in enumerate(self.loss):
            if lf["function"] is not None:
                loss = lf["function"](sr, hr)
                effective_loss = lf["weight"] * loss
                losses.append(effective_loss)
        
        # 측정한 Loss 합하고 저장
        loss_sum = sum(losses)
        
        return loss_sum