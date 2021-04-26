## Step 1
!git clone https://github.com/YoongiKim/CIFAR-10-images

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from imutils import paths

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# train.csv

path = '/content/CIFAR-10-images/train'

# getting all the image paths
image_paths = list(paths.list_images(path))

labels = []
for img_path in image_paths:
  labels.append(img_path.split('/')[4])
data = pd.DataFrame(columns=['image_paths', 'labels'])
data['image_paths'] = image_paths
data['labels'] = labels

data.to_csv('/content/CIFAR-10-images/train.csv', index=False)

# test.csv
path = '/content/CIFAR-10-images/test'

# getting all the image paths
image_paths = list(paths.list_images(path))

labels = []
for img_path in image_paths:
  labels.append(img_path.split('/')[4])
data = pd.DataFrame(columns=['image_paths', 'labels'])
data['image_paths'] = image_paths
data['labels'] = labels

data.to_csv('/content/CIFAR-10-images/test.csv', index=False)

# create dataset class
class MyDataset(Dataset):
    
    def __init__(self, csv_file, transform=None):
        
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx,0]
        image = io.imread(img_path)

        label =  self.data_frame.iloc[idx,1]
        
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}



train_dataset = MyDataset(csv_file='/content/CIFAR-10-images/train.csv')
fig = plt.figure()

for i in range(len(train_dataset)): # __len__() is called internally
    sample = train_dataset[i]       # __getitem__(i) is called internally

    print(i, sample['image'].shape, sample['label'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break

num_workers = 0
batch_size = 20
valid_size = 0.2


num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]


train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
#valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
