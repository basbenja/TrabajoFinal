import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, dropout):
        super(ConvBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, padding=0),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.AvgPool1d(kernel_size=2),

            nn.Flatten()
        )

    def forward(self, x):
        return self.backbone(x)