import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
import tqdm
import random
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import cv2

import albumentations
from albumentations import torch as AT
from pytorchcv.model_provider import get_model as ptcv_get_model

labels = pd.read_csv('data/train_labels.csv')

# print(len(os.listdir("data/train")))
# print(len(os.listdir("data/test")))

# fig = plt.figure(figsize=(25, 4))
# # display 20 images
# train_imgs = os.listdir("data/train")
# for idx, img in enumerate(np.random.choice(train_imgs, 20)):
#     ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
#     im = Image.open("data/train" + img)
#     plt.imshow(im)
#     lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]
#     ax.set_title('Label: %s'%lab)

labels.label.value_counts()
tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.101)
img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}


class CancerDataset(Dataset):
    def __init__(self, datafolder, datatype='train', transform = transforms.Compose([transforms.CenterCrop(64),transforms.ToTensor()]), labels_dict={}):
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
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image=img)
        image = image['image']

        img_name_short = self.image_files_list[idx].split('.')[0]

        if self.datatype == 'train':
            label = self.labels_dict[img_name_short]
        else:
            label = 0
        return image, label


data_transforms = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.HorizontalFlip(),
    albumentations.RandomBrightness(),
    albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(),
    AT.ToTensor()
    ])
data_transforms_test = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(),
    AT.ToTensor()
    ])

dataset = CancerDataset(datafolder='data/train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)
test_set = CancerDataset(datafolder='data/test/', datatype='test', transform=data_transforms_test)
train_sampler = SubsetRandomSampler(list(tr.index))
valid_sampler = SubsetRandomSampler(list(val.index))
batch_size = 64
num_workers = 0
# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

model_conv = ptcv_get_model("cbam_resnet50", pretrained=True)

model_conv.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
model_conv.last_linear = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features=2048, out_features=512, bias=True), nn.SELU(),
                                      nn.Dropout(0.8),  nn.Linear(in_features=512, out_features=1, bias=True))

model_conv.cuda()
# model_conv.load_state_dict(torch.load("model2.pt"))
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model_conv.parameters(), lr=0.0004)

scheduler = StepLR(optimizer, 5, gamma=0.2)
scheduler.step()
val_auc_max = 0
patience = 25
# current number of tests, where validation loss didn't increase
p = 0
# whether training should be stopped
stop = False

# number of epochs to train the model
n_epochs = 10
for epoch in range(1, n_epochs + 1):

    if stop:
        print("Training stop.")
        break

    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    train_auc = []

    for tr_batch_i, (data, target) in enumerate(train_loader):

        model_conv.train()
        # time_s = time.time()

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output[:, 0], target.float())
        train_loss.append(loss.item())

        a = target.data.cpu().numpy()
        try:
            b = output[:, 0].detach().cpu().numpy()
            train_auc.append(roc_auc_score(a, b))
        except:
            pass

        loss.backward()
        optimizer.step()

        # time_e = time.time()
        # delta_t = time_e - time_s
        # print("training.... (time cost:%.3f s)"% delta_t)

        if (tr_batch_i + 1) % 600 == 0:
            model_conv.eval()
            val_loss = []
            val_auc = []
            for val_batch_i, (data, target) in enumerate(valid_loader):
                data, target = data.cuda(), target.cuda()
                output = model_conv(data)

                loss = criterion(output[:, 0], target.float())

                val_loss.append(loss.item())
                a = target.data.cpu().numpy()
                try:
                    b = output[:, 0].detach().cpu().numpy()
                    val_auc.append(roc_auc_score(a, b))
                except:
                    pass

            # print('Epoch %d, batches:%d, train loss: %.4f, valid loss: %.4f.'%(epoch, tr_batch_i, np.mean(train_loss), np.mean(val_loss)))
            print('Epoch %d, batches:%d, train loss: %.4f, valid loss: %.4f.' % (
            epoch, tr_batch_i, np.mean(train_loss), np.mean(val_loss))
                  + '  train auc: %.4f, valid auc: %.4f' % (np.mean(train_auc), np.mean(val_auc)))
            train_loss = []
            train_auc = []
            valid_auc = np.mean(val_auc)
            if valid_auc > val_auc_max:
                print('Validation auc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    val_auc_max,
                    valid_auc))
                # torch.save(model_conv.state_dict(), 'model_val%d.pt'%(valid_auc*10000))
                torch.save(model_conv.state_dict(), 'model_attention.pt')
                val_auc_max = valid_auc
                p = 0
            else:
                p += 1
                if p > patience:
                    print('Early stop training')
                    stop = True
                    break
            scheduler.step()


model_conv.eval()
preds = []
for batch_i, (data, target) in enumerate(test_loader):
    data, target = data.cuda(), target.cuda()
    output = model_conv(data)

    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)
test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})
test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
sub = pd.read_csv('data/sample_submission.csv')
sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')
sub = sub[['id', 'preds']]
sub.columns = ['id', 'label']
sub.head()
sub.to_csv('sub_attention.csv', index=False)