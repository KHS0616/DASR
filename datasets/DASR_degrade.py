import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_sigma(sig_x, sig_y, radians):
    """ anisotropic 가우시안 커널의 시그마 값 생성 함수 """
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma

def anisotropic_gaussian_kernel(batch, kernel_size, covar):
    """ anisotropic 가우시안 커널 생성 함수 """
    # torch.arange(start=0, end, step=1) - 텐서 형식의 순차적인 배열 생성
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2

    # torch.Tensor.repeat- 텐서를 입력 사이즈 만큼 반복해서 복제 생성
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    xy = torch.stack([xx, yy], -1).view(batch, -1, 2)

    # torch.inverse - 역행렬 연산 수행
    # 시그마의 역행렬 측정
    inverse_sigma = torch.inverse(covar)

    # torch.exp(input) - 자연상수의 지수로 사용
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

    # 최종적으로 생성된 커널을 반환, keepdim은 sum연산 후 input shape 유지할 지 결정여부
    return kernel / kernel.sum([1, 2], keepdim=True)


def isotropic_gaussian_kernel(batch, kernel_size, sigma):
    """ isotropic 가우시안 커널 생성 함수 """
    # torch.arange(start=0, end, step=1) - 텐서 형식의 순차적인 배열 생성
    ax = torch.arange(kernel_size).float().cuda() - kernel_size//2

    # torch.Tensor.repeat- 텐서를 입력 사이즈 만큼 반복해서 복제 생성
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)

    # torch.exp(input) - 자연상수의 지수로 사용
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    # 최종적으로 생성된 커널을 반환, keepdim은 sum연산 후 input shape 유지할 지 결정여부
    return kernel / kernel.sum([1,2], keepdim=True)

class Bicubic(nn.Module):
    """ DASR 바이큐빅 모듈 """
    def __init__(self):
        super(Bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        
        # torch.arrange - start 부터 end 까지 1차원 텐서 생성
        # dtype 을 사용하여 type 지정해주자
        # x0 - h, x1 - w
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).cuda()
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        # torch.floor - 정수형으로 내림연산
        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        # numpy ceil - 올림 연산
        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0), torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1), torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)

        # torch.eq - A,B tensor 비교해서 값이 같은지 다른지 비교
        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1/4):
        """ 순전파 """
        # 입력 텐서의 배치사이즈, 채널, 높이, 폭 저장
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0]
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out

class BatchBlur(nn.Module):
    """ DASR 블러 모듈 """
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        # ReflectionPad2d - 거울 패딩
        # 파라미터 값 만큼 너비, 높이 간격을 거울처럼 채움
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.pad = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))

    def forward(self, input, kernel):
        """ 순전파 """
        # 입력 값의 배치, 채널, 높이, 너비를 저장
        B, C, H, W = input.size()

        # 지정된 범위만큼 패딩 및 크기 저장
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))

            # torch.Tensor.contiguous - 메모리상에서 비 연속적인 텐서 값들을 연속적으로 바꿔준다.
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            # 생성한 블러 커널로 합성곱 연산을 수행하여 블러 이미지 생성
            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))

            # torch.Tensor.contiguous - 메모리상에서 비 연속적인 텐서 값들을 연속적으로 바꿔준다.
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            # 생성한 블러 커널로 합성곱 연산을 수행하여 블러 이미지 생성
            return F.conv2d(input_CBHW, kernel, groups=B*C).view((B, C, H, W))

class Gaussin_Kernel(object):
    """ DASR 가우시안 커널 클래스 """
    def __init__(self, kernel_size=21, blur_type='iso_gaussian',
                 sig=2.6, sig_min=0.2, sig_max=4.0,
                 lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0):
        # 옵션 등록
        self.kernel_size = kernel_size
        self.blur_type = "aniso_gaussian" # blur_type

        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.theta = theta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def __call__(self, batch, random):
        # 랜덤한 가우시안 커널 생성
        if random == True:
            if self.blur_type == 'iso_gaussian':
                # 배치사이즈 만큼 랜덤한 텐서를 생성
                # 시그마 값을 이용하여 노이즈 강도 조절
                x = torch.rand(batch).cuda() * (self.sig_max - self.sig_min) + self.sig_min

                # 생성된 텐서를 이용하여 isotropic 가우시안 커널 생성
                k = isotropic_gaussian_kernel(batch, self.kernel_size, x)
                return k

            elif self.blur_type == 'aniso_gaussian':
                # 배치사이즈를 이용하여 theta 값 선언
                theta = torch.rand(batch).cuda() / 180 * math.pi

                # lambda 범위를 이용하여 lambda값 선언
                lambda_1 = torch.rand(batch).cuda() * (self.lambda_max - self.lambda_min) + self.lambda_min
                lambda_2 = torch.rand(batch).cuda() * (self.lambda_max - self.lambda_min) + self.lambda_min

                # 시그마 값 계산
                covar = cal_sigma(lambda_1, lambda_2, theta)

                # anisotropic 가우시안 커널 생성
                kernel = anisotropic_gaussian_kernel(batch, self.kernel_size, covar)
                return kernel

        # 고정된 가우시안 커널 생성
        else:
            if self.blur_type == 'iso_gaussian':
                # 생성된 시그마 값을 이용하여 텐서 생성
                x = torch.ones(1).cuda() * self.sig

                # 생성된 텐서를 이용하여 isotropic 가우시언 커널 생성
                k = isotropic_gaussian_kernel(1, self.kernel_size, x)
                return k
            elif self.blur_type == 'aniso_gaussian':
                # 생성된 theta, lambda 값을 이용하여 텐서 생성
                theta = torch.ones(1).cuda() * self.theta / 180 * math.pi
                lambda_1 = torch.ones(1).cuda() * self.lambda_1
                lambda_2 = torch.ones(1).cuda() * self.lambda_2

                # 시그마 생성 후 anisotropic 가우시안 커널 생성
                covar = cal_sigma(lambda_1, lambda_2, theta)
                kernel = anisotropic_gaussian_kernel(1, self.kernel_size, covar)
                return kernel

class Degradeprocessing(object):
    """ DASR Degread 클래스 """
    def __init__(self, device):
        ## 옵션들
        scale = 2
        mode = "bicubic"
        kernel_size = 21
        blur_type = "iso_gaussian"
        sig = 2.6
        sig_min=0.2
        sig_max=4.0
        lambda_1=0.2
        lambda_2=4.0
        theta=0
        lambda_min=0.2
        lambda_max=4.0
        noise=0.0
        self.device = device

        self.kernel_size = kernel_size
        self.scale = scale
        self.mode = mode
        self.noise = noise

        self.gen_kernel = Gaussin_Kernel(
            kernel_size=kernel_size, blur_type=blur_type,
            sig=sig, sig_min=sig_min, sig_max=sig_max,
            lambda_1=lambda_1, lambda_2=lambda_2, theta=theta, lambda_min=lambda_min, lambda_max=lambda_max
        )
        self.blur = BatchBlur(kernel_size=kernel_size).to(self.device)
        self.bicubic = Bicubic().to(self.device)

    def __call__(self, hr_tensor, random=True):
        with torch.no_grad():
            # 가우시안 커널 적용하지 않고 shape만 변환
            if self.gen_kernel.blur_type == 'iso_gaussian' and self.gen_kernel.sig == 0:
                B, N, C, H, W = hr_tensor.size()
                hr_blured = hr_tensor.view(-1, C, H, W)
                b_kernels = None

            # 가우시안 커널, 블러 적용
            else:
                B, N, C, H, W = hr_tensor.size()
                b_kernels = self.gen_kernel(B, random)  # B degradations

                # blur
                hr_blured = self.blur(hr_tensor.view(B, -1, H, W), b_kernels)
                hr_blured = hr_blured.view(-1, C, H, W)  # BN, C, H, W

            # Downsample 적용
            if self.mode == 'bicubic':
                lr_blured = self.bicubic(hr_blured, scale=1/self.scale)
            elif self.mode == 's-fold':
                lr_blured = hr_blured.view(-1, C, H//self.scale, self.scale, W//self.scale, self.scale)[:, :, :, 0, :, 0]

            # 랜덤 노이즈 추가
            if self.noise > 0:
                _, C, H_lr, W_lr = lr_blured.size()
                noise_level = torch.rand(B, 1, 1, 1, 1).to(lr_blured.device) * self.noise if random else self.noise
                noise = torch.randn_like(lr_blured).view(-1, N, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
                lr_blured.add_(noise)

            # 텐서 값을 0~255 범위로 조절
            lr_blured = torch.clamp(lr_blured.round(), 0, 255)

            # Degrade 적용된 이미지 반환
            return lr_blured.view(B, N, C, H//int(self.scale), W//int(self.scale)), b_kernels