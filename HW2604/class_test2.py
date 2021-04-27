
!git clone https://github.com/YoongiKim/CIFAR-10-images		#Download dataset

import pandas as pd
from imutils import paths
from sklearn import preprocessing

import numpy as np
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def makecsv(data_path, csv_path):
  # getting all the image paths
  image_paths = list(paths.list_images(data_path))
  #get all labels
  labels = []
  for img_path in image_paths : labels.append(img_path.split('/')[4])
  #String labels to numeric label
  le=preprocessing.LabelEncoder()
  labels = le.fit_transform(labels)
  #make a dataframe
  data = pd.DataFrame(columns=['image_paths', 'labels'])
  data['image_paths'] = image_paths
  data['labels'] = labels
  # load to csv file
  data.to_csv(csv_path, index=False)

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': torch.as_tensor(label)}
        
class Normalize(object):
    def __init__(self, meanstd):
        self.mean = meanstd[0]
        self.std = meanstd[1]
        
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img/255
        
        for t, m, s in zip(img, self.mean, self.std) : t.sub_(m).div_(s)

        return {'image': img, 'label': label}

class MyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx, 0]
        image = io.imread(img_path)
        label =  self.data_frame.iloc[idx, 1]
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def dataloader(traincsv, testcsv, batch_size, valid_size):
    train_dataset = MyDataset(csv_file = traincsv, transform = transforms.Compose([ ToTensor(), Normalize(((0.5,0.5,0.5),(0.5,0.5,0.5))) ]))
    test_dataset = MyDataset(csv_file = testcsv, transform=transforms.Compose([ToTensor(), Normalize(((0.5,0.5,0.5),(0.5,0.5,0.5)))]))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

# Visualize a Batch of Data
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def data_visualize(load_data):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    data = next(iter(load_data))
    images, labels = data['image'], data['label']
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])


class CNN(nn.Module):
    def __init__(self, ip, n_hidden_layers, n_output ):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(ip, n_hidden_layers[0], kernel_size=3, padding=1)])

        for i in range (len(n_hidden_layers[:-1])):
            self.layers.extend([ nn.Conv2d(n_hidden_layers[i], n_hidden_layers[i+1], kernel_size=3, padding=1) ])
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, n_output)

        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.layers[0](x)))
        x = self.pool(F.relu(self.layers[1](x)))
        x = self.pool(F.relu(self.layers[2](x)))
        
        x = x.view(-1, 64 * 4 * 4)
        
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        
        x = self.fc3(self.dropout(x))
        return x

def train(epochs, save_path, train_loader, valid_loader, criterion, optimizer ):
    valid_loss_min = np.Inf
    
    for e in range(epochs):
        train_loss = 0
        valid_loss = 0
        
        model.train()
        for i_batch, data in enumerate(train_loader):
            images, labels = data['image'].cuda(), data['label'].cuda()
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(images)
        else:
            model.eval()
            with torch.no_grad():
                for i_batch, data in enumerate(valid_loader):
                    images, labels = data['image'].cuda(), data['label'].cuda()
                    output = model(images)
                    loss = criterion(output, labels)
                    valid_loss += loss.item()*len(images)

        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
       
        print("epoch: {:2d}/{}".format(e+1, epochs), "train_loss: {:.3f}".format(train_loss), "valid_loss: {:.3f}".format(valid_loss))
        
        # saving the model
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print("Model Save")
            valid_loss_min = valid_loss

def test(model_path, test_loader, criterion):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    model.load_state_dict(torch.load(model_path))

    test_loss = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    model.eval()
    for i_batch, data in enumerate(test_loader):
        images, labels = data['image'].cuda(), data['label'].cuda()
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()*images.size(0)

        _, pred = torch.max(output, 1)    
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())

        for i in range(20):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    pf = open("/content/performance.txt", "a")
    pf.writelines('Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                  classes[i], 100 * class_correct[i] / class_total[i],
                  np.sum(class_correct[i]), np.sum(class_total[i])))
            pf.writelines('\nTest Accuracy of %5s: %2d%% (%2d/%2d)' % (
                          classes[i], 100 * class_correct[i] / class_total[i],
                          np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
            pf.writelines('\nTest Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    pf.writelines('\n\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
                  100. * np.sum(class_correct) / np.sum(class_total),
                  np.sum(class_correct), np.sum(class_total)))
    pf.close()



if __name__ == "__main__":
  traindata_path = '/content/CIFAR-10-images/train'
  testdata_path = '/content/CIFAR-10-images/test'
  traincsv_path = '/content/CIFAR-10-images/train.csv'
  testcsv_path = '/content/CIFAR-10-images/test.csv'
  
  makecsv(traindata_path, traincsv_path)          #make train.csv
  makecsv(testdata_path, testcsv_path)            #make test.csv

  train_loader, valid_loader, test_loader = dataloader(traincsv_path, testcsv_path, 20, 0.2)

  data_visualize(train_loader)                   # show one batch of train data
  data_visualize(test_loader)                    # show one batch of train data

  model = CNN(3,[16,32,64],10)                   # create model
  model.cuda()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01)

  epochs = 30
  model_path = '/content/model.pt'
  train(epochs, model_path, train_loader, valid_loader, criterion, optimizer)   # train the model

  test(model_path, test_loader, criterion)      # test the model and value store in performance.txt



