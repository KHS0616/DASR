"""
DASR
"""
import torch
import torch.nn as nn
from torch.nn.functional import embedding


class BaseEncoder(nn.Module):
    """ Degradation Representation Learning Module Base Encoder """
    def __init__(self):
        super(BaseEncoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out

class DegradationEncoder(nn.Module):
    def __init__(self, training=True):
        super(DegradationEncoder, self).__init__()
        # Encoder 하이퍼 파라미터 설정
        # K - queue size, number of negative keys (default: 6554)
        # m - momentum of update key encoder (default : 0.999)
        # T - softmax temperature (default : 0.07)
        dim = 256
        self.K = 32*256
        self.m = 0.999
        self.T = 0.07

        self.training = training

        # queue, key encoder 지정
        self.encoder_q = BaseEncoder()
        self.encoder_k = BaseEncoder()

        # queue, key encoder weight 값을 동일하게 초기화 한다.
        # key encoder weight 값은 변동시키지 않는다.
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        # register_buffer - 중간에 torch tensor layer를 저장하는데 사용, weight 변동 X
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """ Momentum update of the key encoder """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """ 순전파 """
        # im_q - a batch of query images
        # im_k - a batch of key images
        # 학습상태일 경우 학습에 필요한 데이터 전부 반환
        if self.training:
            # query feature 측정 및 정규화
            # queries : NxC
            embedding, q = self.encoder_q(im_q)
            q = nn.functional.normalize(q, dim=1)

            # key feature 측정 및 정규화
            # keys : NxC
            with torch.no_grad():
                # key encoder 업데이트
                self._momentum_update_key_encoder()

                # key 측정
                _, k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)

            # logits 측정, 아인슈타인 계산법 사용
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # 최종 logits, Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            return embedding, logits, labels

        else:
            embedding, _ = self.encoder_q(im_q)
            return embedding
