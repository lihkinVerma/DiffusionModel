
import argparse
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from dataset import CifarDataset

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
        t = t.unsqueeze(-1).type(torch.float)
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
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256, num_classes = 10, device = torch.device("cpu")):
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

class Diffusion(nn.Module):
    def __init__(self, args):
        super(Diffusion, self).__init__()
        self.epochs = args.epochs
        self.lr = args.lr
        self.img_size = args.img_size
        self.device = args.device
        self.steps = 1000
        self.beta_start = 1e-5
        self.beta_end = 1e-2
        self.beta = self.get_beta_schedule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim = -1)
        self.model = Unet_conditional(device = self.device)
        self.optimizer, self.lr_schedular, self.scaler = self.get_model_settings()

    def get_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.steps).to(self.device)

    def sample_timesteps(self, num_steps):
        return torch.randint(1, self.steps, (num_steps,)).to(self.device)

    def add_noise_to_images(self, batch_images, sample_timesteps):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[sample_timesteps])[:, None, None, None]
        sqrt_on_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[sample_timesteps])[:, None, None, None]
        noise = torch.rand_like(batch_images)
        noised_images = sqrt_alpha_bar * batch_images + sqrt_on_minus_alpha_bar * noise
        return noised_images, noise

    def get_model_settings(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode ='min', factor=0.1, patience=2)
        scaler = GradScaler()
        return optimizer, lr_schedular, scaler

    def forward(self, batch_images, batch_labels):
        sample_timesteps = self.sample_timesteps(batch_images.shape[0])
        noised_images, actual_noise = self.add_noise_to_images(batch_images, sample_timesteps)
        predicted_noise = self.model(noised_images, sample_timesteps, batch_labels)
        return actual_noise, predicted_noise

    def calculate_loss(self, actual_noise, predicted_noise):
        mse = nn.MSELoss()
        return mse(actual_noise, predicted_noise)

    def fit(self, epoch, dataloader, type = "train"):
        if type == "train":
            self.train()
        else:
            self.eval()
        losses = []
        for idx, batch in enumerate(tqdm(dataloader)):
            images, labels = batch
            images.to(self.device)
            labels.to(self.device)
            with autocast():
                actual_noise, predicted_noise = self.__call__(images, labels)
                loss = self.calculate_loss(actual_noise, predicted_noise)
                losses.append(loss)
            if type == "train":
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        if type == "train":
            self.lr_schedular.step(loss)
        return torch.tensor(losses).mean()

    def sample_new_images(self, num_images_to_generate, guiding_labels, guidance_scale = 3):
        self.model.eval()
        with torch.no_grad():
            random_images = torch.randn((num_images_to_generate, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.steps)), position=0):
                t = (torch.ones(num_images_to_generate) * i).long().to(self.device)
                predicted_noise = self.model(random_images, t, guiding_labels)
                if guidance_scale > 0:
                    unconditioned_predicted_noise = self.model(random_images, t, None)
                    predicted_noise = torch.lerp(unconditioned_predicted_noise, predicted_noise, guidance_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.rand_like(random_images)
                else:
                    noise = torch.zeros_like(random_images)
                random_images = ( 1 / torch.sqrt(alpha) ) * (random_images - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + beta * noise
        self.model.train()
        clear_images = (random_images.clamp(-1, 1) + 1) / 2
        clear_images = (clear_images * 255).type(torch.uint8)
        return clear_images

def get_args():
    diffusion_parser = argparse.ArgumentParser()
    diffusion_parser.add_argument("--name", type=str, default="nk_diffusion",
                                  help="Give name to this run of Diffusion Model")
    diffusion_parser.add_argument("--dataset_path", type=str, default='./data')
    diffusion_parser.add_argument("--epochs", type=int, default=10)
    diffusion_parser.add_argument("--batch_size", type=int, default=128)
    diffusion_parser.add_argument("--img_size", type=int, default=32)
    diffusion_parser.add_argument("--num_classes", type=int, default=10)
    diffusion_parser.add_argument("--device",
                                  default=torch.device(
                                      "cuda" if torch.cuda.is_available() else "cpu"))
    diffusion_parser.add_argument("--lr", type=float, default=1e-5)
    args = diffusion_parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    val_dataset = CifarDataset(args.dataset_path, "valid")
    val_dataloader = val_dataset.get_dataloader(args.batch_size)
    diffusion_model = Diffusion(args)
    loss = diffusion_model.fit(0, val_dataloader, "train")
