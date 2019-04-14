import pandas as pd
import numpy as np
import os, time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
train_on_gpu = True

from PIL import Image


import torch.nn as nn
import torchvision.transforms as transforms

labels = pd.read_csv('data/train_labels.csv')

# print(f'{len(os.listdir("../input/train"))} pictures in train.')
# print(f'{len(os.listdir("../input/test"))} pictures in test.')

# fig = plt.figure(figsize=(25, 4))
# # display 20 images
# train_imgs = os.listdir("data/train")
# for idx, img in enumerate(np.random.choice(train_imgs, 20)):
#     ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
#     im = Image.open("data/train/" + img)
#     plt.imshow(im)
#     lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]
#     ax.set_title(f'Label: {lab}')
# plt.show()

labels.label.value_counts()

data_transforms = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
data_transforms_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.1)

# print(tr,val)
img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}


class CancerDataset():
    def __init__(self, datafolder, datatype='train', transform = transforms.Compose([transforms.CenterCrop(32),transforms.ToTensor()]), labels_dict={}):
        self.datafolder = datafolder
        self.datatype = datatype
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform
        self.labels_dict = labels_dict
        if self.datatype == 'train':
            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]
        else:
            self.labels = [0 for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder, self.image_files_list[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        img_name_short = self.image_files_list[idx].split('.')[0]

        if self.datatype == 'train':
            label = self.labels_dict[img_name_short]
        else:
            label = 0
        return image, label


# Load train data
dataset = CancerDataset(datafolder='data/train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)
# Load test data
test_set = CancerDataset(datafolder='data/test/', datatype='test', transform=data_transforms_test)

train_sampler = SubsetRandomSampler(list(tr.index))
valid_sampler = SubsetRandomSampler(list(val.index))
batch_size = 5120
num_workers = 0
# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)


# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


model_conv = Net()
model_conv.cuda()
criterion = nn.BCELoss()

optimizer = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

valid_loss_min = np.Inf
patience = 7
# current number of epochs, where validation loss didn't increase
p = 0
# whether training should be stopped
stop = False

# number of epochs to train the model
n_epochs = 100
for epoch in range(1, n_epochs+1):
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    exp_lr_scheduler.step()
    train_auc = []

    for batch_i, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output[:,1], target.float())
        train_loss.append(loss.item())

        a = target.data.cpu().numpy()
        b = output[:,-1].detach().cpu().numpy()
        train_auc.append(roc_auc_score(a, b))

        loss.backward()
        optimizer.step()

    model_conv.eval()
    val_loss = []
    val_auc = []
    for batch_i, (data, target) in enumerate(valid_loader):
        data, target = data.cuda(), target.cuda()
        output = model_conv(data)

        loss = criterion(output[:,1], target.float())

        val_loss.append(loss.item())
        a = target.data.cpu().numpy()
        b = output[:,-1].detach().cpu().numpy()
        val_auc.append(roc_auc_score(a, b))

    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train auc: {np.mean(train_auc):.4f}, valid acc: {np.mean(val_auc):.4f}')

    valid_loss = np.mean(val_loss)
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model_conv.state_dict(), 'model.pt')
        valid_loss_min = valid_loss
        p = 0

    # check if validation loss didn't improve
    if valid_loss > valid_loss_min:
        p += 1
        print(f'{p} epochs of increasing val loss')
        if p > patience:
            print('Stopping training')
            stop = True
            break

    if stop:
        break