
import os
import torch
import argparse
from model import Diffusion
from dataset import CifarDataset
from utils import plot_images, save_images

def generate_new_samples(args):
    val_dataset = CifarDataset(args.dataset_path, "valid")
    val_dataloader = val_dataset.get_dataloader(args.batch_size)
    diffusionModel = Diffusion(args)
    checkpoint_path = os.path.join("models", args.name, "checkpoint.pt")
    diffusionModel.load_state_dict(torch.load(checkpoint_path))
    for idx, batch in enumerate(val_dataloader):
        images, labels = batch
        images = images.to(args.device)
        labels = labels.to(args.device)
        sampled_images = diffusionModel.sample_new_images(images.shape[0], labels, args.guidance_scale)
        plot_images(sampled_images)
        save_images(sampled_images, os.path.join("results", args.name, f"{idx}.jpg"))
    return True

def get_args():
    diffusion_parser = argparse.ArgumentParser()
    diffusion_parser.add_argument("--name", type=str, default="nk_diffusion",
                                  help="Give name to this run of Diffusion Model")
    diffusion_parser.add_argument("--dataset_path", type=str, default='./data')
    diffusion_parser.add_argument("--epochs", type=int, default=10)
    diffusion_parser.add_argument("--batch_size", type=int, default=64)
    diffusion_parser.add_argument("--img_size", type=int, default=32)
    diffusion_parser.add_argument("--num_classes", type=int, default=10)
    diffusion_parser.add_argument("--device",
                                  default=torch.device(
                                      "cuda" if torch.cuda.is_available() else "cpu"))
    diffusion_parser.add_argument("--lr", type=float, default=1e-5)
    diffusion_parser.add_argument("--guidance_scale", type=int, default=3)
    args = diffusion_parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    generate_new_samples(args)
