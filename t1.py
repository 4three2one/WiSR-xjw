import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1)
        channel_att = self.fc(avg_out).unsqueeze(2)
        channel_att = self.sigmoid(channel_att)
        return x * channel_att

if __name__ == '__main__':


    model=ChannelAttention(128)
    x=torch.randn(89,128,50)
    y=model(x)
    pass