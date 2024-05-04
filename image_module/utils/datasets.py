import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from .data_constants import (IMAGENET_DEFAULT_MEAN,
                             IMAGENET_DEFAULT_STD)

from .dataset_folder import MultiTaskImageFolder


class DataAugmentationPretrain(object):
    def __init__(self, args):
        mean = IIMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.transform = transform_train

    def __call__(self, task_dict):
        for task in task_dict:
            task_dict[task] = self.transform(task_dict[task])
        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationPretrain,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationPretrain(args)
    return MultiTaskImageFolder(args.data_path, args.all_domains, transform=transform)


