
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0
    
    def update_model_average(self, ma_model, current_model):
        for cur_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, cur_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + new * (1-self.beta)

    def reset_parameter(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

    def step_ema(self, ema_model, model, step_start_ema = 2000):
        if self.step < step_start_ema:
            self.reset_parameter(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first = True)
        self.ln = nn.LayerNorm([channels])
        self.ff_net = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
    
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1,2)
        x_ln = self.ln(x)
        attention_val,_ = self.mha(x_ln, x_ln, x_ln)
        attention_val = attention_val + x
        attention_val = self.ff_net(attention_val) + attention_val
        return attention_val.swapaxes(2,1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias = False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias= False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv_net(x))
        return self.double_conv_net(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode = "bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels),
            DoubleConv(in_channels, out_channels, in_channels//2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim = 1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Unet(nn.Module):
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256, device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(256, 64)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(128, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_enc(self, t, channels):
        inv_freq = 1.0/(
            10000 ** (torch.arange(0, channels, 2, device = self.device).float() / channels)
        )
        pos_enc_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim = -1)
        return pos_enc

    def forward(self, x, t):
        t.unsqueeze(-1).type(torch.float)
        t = self.pos_enc(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t))
        x3 = self.sa2(self.down2(x2, t))
        x4 = self.sa3(self.down3(x3, t))

        x4 = self.bot3(self.bot2(self.bot1(x4)))

        x5 = self.sa4(self.up1(x4, x3, t))
        x6 = self.sa5(self.up2(x5, x2, t))
        x7 = self.sa6(self.up3(x6, x1, t))
        o = self.outc(x7)
        return o

class Unet_conditional(nn.Module):
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256, num_classes = None, device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 16)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 8)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 4)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 8)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 16)
        self.up3 = Up(128, 32)
        self.sa6 = SelfAttention(32, 32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=time_dim)

    def pos_enc(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t.unsqueeze_(-1).type(torch.float)
        t = self.pos_enc(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t))
        x3 = self.sa2(self.down2(x2, t))
        x4 = self.sa3(self.down3(x3, t))

        x4 = self.bot3(self.bot2(self.bot1(x4)))

        x5 = self.sa4(self.up1(x4, x3, t))
        x6 = self.sa5(self.up2(x5, x2, t))
        x7 = self.sa6(self.up3(x6, x1, t))
        o = self.outc(x7)
        return o

if __name__ == "__main__":
    net = Unet_conditional(num_classes = 10, device = torch.device("cpu"))
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(5, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)

# Reference - https://github.com/dome272/Diffusion-Models-pytorch