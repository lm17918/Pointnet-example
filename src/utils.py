import os
import random

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pointnetloss:
    def __init__(self, alpha=0.0001):
        self.alpha = alpha
        self.criterion = torch.nn.NLLLoss()

    def forward(self, outputs, labels, m3x3, m64x64):
        bs = outputs.size(0)
        self.id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        self.id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
        self.id3x3 = self.id3x3.to(device)
        self.id64x64 = self.id64x64.to(device)

        diff3x3 = self.id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        diff64x64 = self.id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        return self.criterion(outputs, labels) + self.alpha * (
            torch.norm(diff3x3) + torch.norm(diff64x64)
        ) / float(bs)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
