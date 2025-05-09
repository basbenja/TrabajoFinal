import torch
import torch.nn as nn

class FCBlock(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super(FCBlock, self).__init__()
        backbone = []
        for hidden_size in hidden_sizes:
            backbone.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = hidden_size
        backbone.append(nn.Linear(hidden_sizes[-1], 1))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        logits = self.backbone(x)
        return logits