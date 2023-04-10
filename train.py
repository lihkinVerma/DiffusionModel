
import os
import torch
from dataset import CifarDataset
from model import Diffusion

def train_diffusion_model(args):
    train_dataset = CifarDataset(args.dataset_path, "train")
    val_dataset = CifarDataset(args.dataset_path, "valid")
    train_dataloader = train_dataset.get_dataloader(args.batch_size)
    val_dataloader = val_dataset.get_dataloader(args.batch_size)
    diffusion_model = Diffusion(args)
    best_loss = torch.inf
    for epoch in range(args.epochs):
        train_loss = diffusion_model.fit(epoch, train_dataloader, "train")
        with torch.no_grad():
            val_loss = diffusion_model.fit(epoch, val_dataloader, "valid")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(diffusion_model.state_dict(),
                       os.path.join("models", args.name, "checkpoint.pt"))
    return True
