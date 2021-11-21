from numpy.lib.type_check import imag
import torch

import os
import PIL.Image as plt
import imageio

from .transformer import makeTransform, augment, get_patch, set_channel, np2Tensor

def check_image(file_name):
    """ 파일이 이미지 형식인지 확인하는 함수 """
    ext = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")
    if file_name.endswith(ext):
        return True
    else:
        return False

def preprocesses(img):
    """ DASR 전처리 함수 """
    # HR 리스트 선언
    HR = []

    # Augment 설정
    opt = {
        "hflip" : True,
        "vflip" : True,
        "rot" : True,
        "crop" : {
            "status" : False,
            "size" : 96
        },
        "resize" : {
            "status" : False,
            "size" : 48
        },
        "tensor" : False
    }
    augment_transformer = makeTransform(opt)

    # Augment 실행
    img = augment_transformer(img)

    # patch 제작 함수 생성
    opt["hflip"] = False
    opt["vflip"] = False
    opt["rot"] = False
    opt["crop"]["status"] = True
    patch_transformer = makeTransform(opt)

    # 텐서 변환 함수 생성
    opt["crop"]["status"] = False
    opt["tensor"] = True
    tensor_transformer = makeTransform(opt)

    for _ in range(2):
        hr_patch = patch_transformer(img)
        HR.append(tensor_transformer(hr_patch))

    return HR

class MultiScaleSRDataset(torch.utils.data.Dataset):
    """ DASR MultiScale 데이터 셋 클래스 """
    def __init__(self):
        # 옵션 값을 변수에 할당
        # self.opt = opt

        # 이미지 리스트 불러오기
        self.image_path = "/workspace/Image/DIV2K+Flickr2K"
        self.img_list = [x for x in os.listdir(self.image_path) if check_image(x)]

        self.train = True

    def __getitem__(self, idx):
        # PIL 형식으로 이미지 불러오기
        image = imageio.imread(os.path.join(self.image_path, self.img_list[idx]))
        imgs = self.get_patch(image)
        hr = [set_channel(img, n_channels=3) for img in imgs]
        hr_tensor = [np2Tensor(img, rgb_range=255) for img in hr]
        # image = plt.open(os.path.join(self.image_path, self.img_list[idx])).convert("RGB")

        # HR 이미지 생성
        # HR = preprocesses(image)
        # return torch.stack(HR, 0)
        return torch.stack(hr_tensor, 0)

    def __len__(self):
        return len(self.img_list)

    def get_patch(self, hr):
        scale = 2
        if self.train:
            out = []
            hr = augment(hr)
            # extract two patches from each image
            for _ in range(2):
                hr_patch = get_patch(
                    hr,
                    patch_size=48,
                    scale=scale
                )
                out.append(hr_patch)
        else:
            out = [hr]
        return out