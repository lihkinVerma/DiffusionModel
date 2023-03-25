
import os
import copy
import torch
import wandb
from torch import nn 
from torch import optim
from tqdm import tqdm
from utils import *
from modules import Unet, EMA, Unet_conditional
import logging
import argparse
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

wandb.init(project = "diffusion-for-images")
# wandb.config.update(args)
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S", filename = "ddpm_logs.log")

class Diffusion:
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, img_size = 32, device = None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.create_noising_schedule().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = -1)

    def create_noising_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def add_noise_to_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # unsqueeze in last dimensions
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        e = torch.rand_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timestep(self, n):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))

    def sample(self, model, n, labels, cfg_scale = 3):
        logging.info(f"Sampling {n} new images")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale>0:
                    uncond_pred_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_pred_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i>1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (x - ((1 - alpha) / torch.sqrt(1-alpha_hat)) * predicted_noise) / torch.sqrt(alpha) + beta * noise
        model.train()
        x = (x.clamp(-1, 1) + 1)/2
        x = (x * 255).type(torch.uint8)
        return x

def train_and_sample(args):
    setup_logging(args.run_name)
    device = args.device
    cifar10 =  datasets.CIFAR10(args.dataset_path, train=True, download=True, transform=transforms.ToTensor())
    cifar10_val = datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset = cifar10, batch_size=args.batch_size) # get_data(args)
    model = Unet_conditional(num_classes= args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    loss_func = nn.MSELoss()
    diffusion = Diffusion(img_size= args.img_size, device= device)
    logger = logging.getLogger(args.run_name)

    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise_to_image(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = loss_func(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE = loss.item()) 
            wandb.log({'step': epoch*l + i, 'loss' : loss.item()})
            logger.info(f"MSE: loss.item() = {loss.item()}, global_step = {epoch*l + i}")
        
        if epoch % 1 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"checkpoint.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_checkpoint.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

    # sampled_images = diffusion.sample(model, images.shape[0])
    # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch.jpg}"))
    # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"checkpoint.pt"))

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM"
    args.epochs = 10
    args.batch_size = 128
    args.img_size = 32
    args.num_classes = 10
    args.dataset_path = r"dataset/"
    args.device = torch.device("cpu") # mps
    args.lr = 1e-4
    train_and_sample(args)

if __name__ == "__main__":
    launch()

# Reference - https://github.com/dome272/Diffusion-Models-pytorch