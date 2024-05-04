import os
# import random
import pandas as pd
# import numpy as np
# import torch
# import torchvision.transforms.functional as TF
from torchvision import transforms
from .dataset_folder_finetune import MultiTaskImageFolderFromCSV


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    csv_data = pd.read_csv(os.path.join(args.csv_path, f'{is_train}.csv'), dtype={args.label_column: int})
    return MultiTaskImageFolderFromCSV(args.data_path, args.in_domains, csv_data, transform=transform, nb_classes=args.nb_classes, label_column=args.label_column)


def build_transform(is_train, args):
    trainsize = (args.input_size, args.input_size)
    if is_train == 'train':
        transform = transforms.Compose([
            transforms.Resize(trainsize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(),
            # transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            # CropCenterSquare(),
            transforms.Resize(trainsize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
