import torch
from torch.nn.modules.activation import Softmax
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import CrossMapLRN2d
from torch.nn.modules.pooling import MaxPool2d
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import pandas as pd
import os
from matplotlib import pyplot as plt

dirname = os.path.dirname(__file__)

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256,256))])
train_ds = torchvision.datasets.ImageFolder(root=os.path.join(dirname, 'dataset/train'), transform=transform)
val_ds = torchvision.datasets.ImageFolder(root=os.path.join(dirname, 'dataset/valid'), transform=transform)
test_ds = torchvision.datasets.ImageFolder(root=os.path.join(dirname, 'dataset/test'), transform=transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=True)



model = nn.Sequential(
    nn.Conv2d(3, 6, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(6, 12, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(12, 24, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(24, 48, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Flatten(),

    nn.Linear(48*16*16, 128),
    nn.Linear(128, 64),
    nn.Linear(64, 2),
    #nn.Softmax()

)

model.cuda()

opt = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

epochs = 10

# img, lab = iter(train_loader).next()
# img = img.cuda()
# lab = lab.cuda()
#print(model(img).argmax(1))
# print(lab.size())
# img = torch.squeeze(img)
# print(img.size())
# plt.imshow( img.permute(1, 2, 0))
# plt.show()


for e in range(epochs):
    correct = 0 
    val_correct = 0
    model.train()
    for i, (img, lab) in enumerate(train_loader):
        img = img.cuda()
        lab = lab.cuda()

        output = model(img)
        loss = loss_fn(output, lab)

        opt.zero_grad()
        loss.backward()
        opt.step()

        correct += torch.sum(output.argmax(1) == lab)

    
    train_acc = correct/len(train_ds)
    print("Epoch:{}, Train Loss: {}, Train Accuracy:{}".format(e, loss.data, train_acc))

        # if(i%100==0):
        #     print("Epoch:{}, Iter:{}, Loss: {}".format(e, i, loss.data))
    
    model.eval()
    with torch.no_grad():
        for j, (img, lab) in enumerate(val_loader):
            img = img.cuda()
            lab = lab.cuda()

            output = model(img)
            val_loss = loss_fn(output, lab)

            val_correct += torch.sum(output.argmax(1) == lab)
        
        val_acc = val_correct/len(val_ds)

        print("Validation Loss: {}, Validation Accuracy:{}".format(val_loss.data, val_acc))




# class FaceMaskDetection(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1)


#     def forward(self, x):
#         nn.Sequential()
