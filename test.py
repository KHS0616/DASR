"""
DASR Test Code

Writer : KHS0616
Last Update : 2021-10-27
"""

import torch

from models.DASR_generator import DASR

import os
import imageio, cv2
import numpy as np

class Tester():
    def __init__(self):
        self.setEnviron()
        self.setData()
        self.setModel()

    def setEnviron(self):
        """ 테스트, 추론 환경설정 메소드 """
        # 이미지 디렉토리 경로 설정
        self.input_dir = "./inputs"
        self.output_dir = "./outputs"

        # Device 설정
        self.device = torch.device("cuda:0")

    def checkImage(self, img):
        """ 이미지 여부 확인 메소드 """
        ext = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")
        return True if img.endswith(ext) else False

    def quantize(self, img, rgb_range):
        """ 이미지 양자화 메소드 """
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    def setData(self):
        """ 데이터 설정 메소드 """
        self.data_list = [x for x in os.listdir(self.input_dir) if self.checkImage(x)]

    def setModel(self):
        """ 테스트, 추론에 사용되는 모델 설정 메소드 """
        # 모델 객체 변수 생성
        self.model = DASR(training=False).to(self.device)

        self.model.load_state_dict(torch.load("./pretrainedModel/DASR_600.pth"))

    def test(self):
        """ 테스트, 추론 메소드 """
        for img_name in self.data_list:
            img = imageio.imread(os.path.join(self.input_dir, img_name))
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))
            img = torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)

            sr = self.model(img[:, 0, ...])
            sr = self.quantize(sr, 255.0)

            sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
            sr = sr[:, :, [2, 1, 0]]
            cv2.imwrite(os.path.join(self.output_dir, img_name), sr)

if __name__ == '__main__':
    tester = Tester()
    tester.test()