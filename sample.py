
import os
import torch
from model import Diffusion
from dataset import CifarDataset
from utils import plot_images, save_images

def generate_new_samples(args):
    val_dataset = CifarDataset(args.dataset_path, "valid")
    val_dataloader = val_dataset.get_dataloader(args.batch_size)
    diffusionModel = Diffusion(args)
    checkpoint_path = os.path.join("models", args.name, "checkpoint.pt")
    diffusionModel.load_state_dict(checkpoint_path)
    for idx, batch in enumerate(val_dataloader):
        images, labels = batch
        sampled_images = diffusionModel.sample_new_images(images.shape[0], labels, args.guidance_scale)
        plot_images(sampled_images)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{idx}.jpg"))
    return True
