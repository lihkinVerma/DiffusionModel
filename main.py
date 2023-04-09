
import torch
import argparse
import train, evaluate

def main(args):
    train.train_diffusion_model(args)
    evaluate.evaluate_diffusion_model(args)
    print("Called main")

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
    args = diffusion_parser.parse_args()
    return args

if __name__ ==  "__main__":
    args = get_args()
    main(args)
