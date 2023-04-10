# DiffusionModel
An easy and straightforward implementation of Diffusion Model for Image Generation

## Dataset 
For this experiment I am using CIFAR-10 dataset, which gets downloaded in the ```dataset.py``` script

### Run commands
1. ```mkdir model```
2. ```cd model```
3. Make directory by name of your run say ```mkdir nk_diffusion```
4. ```cd ../..```
5. ```mkdir results```
6. ```cd results```
7. Make directory by name of your run say ```mkdir nk_diffusion```
8. ```cd ../..```

## Model training
To train and sample from the model, run the command(with or without specific parameters)
```
python3 main.py
```

#### Parameters List
```
1. diffusion_parser.add_argument("--name", type=str, default="nk_diffusion",
                                  help="Give name to this run of Diffusion Model")
2. diffusion_parser.add_argument("--dataset_path", type=str, default='./data')
3. diffusion_parser.add_argument("--epochs", type=int, default=10)
4. diffusion_parser.add_argument("--batch_size", type=int, default=64)
5. diffusion_parser.add_argument("--img_size", type=int, default=32)
6. diffusion_parser.add_argument("--num_classes", type=int, default=10)
7. diffusion_parser.add_argument("--device",
                                  default=torch.device(
                                      "cuda" if torch.cuda.is_available() else "cpu"))
8. diffusion_parser.add_argument("--lr", type=float, default=1e-5)
9. diffusion_parser.add_argument("--guidance_scale", type=int, default=3)
```
You might need to make some changes, if you are not running this code on CUDA enabled device in ```model.py``` file.
The parameters are not tuned to generate best out of the dataset.

## Presentation
This repo also contains a presentation delivered during the Technology Upskilling Series hosted by The Department of Computer Science at University of Toronto with the Vector Institute for AI on 25th March 2023.

## Follow ups
Visit my [blogging channel](https://lih-verma.medium.com) to read more around Diffusion and other advancements in the space of Deep Learning.
