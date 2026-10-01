import os
import cv2
import random
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import config

def random_labelling_pretext(images):
    x, y = [], []
    random.seed(config.random_seed_list[config.random_seed_counter])
    for img in images:
        tasks_labels = np.array([random.randint(0,config.num_classes-1) for _ in range(config.num_tasks)])
        x.append(img)
        y.append(tasks_labels)
        
    config.random_seed_counter += 1
    return np.array(x), np.array(y)

class load_dataset(Dataset):
    def __init__(self, x, y):
        self.x = self.pre_process_images(x)
        self.y = self.pre_process_labels(y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def pre_process_images(self, x):
        x = np.array(x).astype('float32')
        x = torch.Tensor(x).permute(0, 3, 1, 2)
        return x
    
    def pre_process_labels(self, y):
        y = np.array(y).astype('float32')
        return torch.Tensor(y)
