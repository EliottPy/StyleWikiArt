import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from datetime import datetime
GPUbool = torch.cuda.is_available()
if GPUbool:
    device = torch.device('cuda') #'mps' pour un mac m1

numepochs= 10
batch_size = 32
learning_rate = 0.01

transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 40, 3)
        self.conv2 = nn.Conv2d(40, 40, 4)
        self.conv3 = nn.Conv2d(40, 40, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(40*24*24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14) 
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 40*24*24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

batch_size = 32 

train_set =  ImageFolder(root="./db/train_resnet18" , transform=transform)
test_set =  ImageFolder(root="./db/valid_resnet18" , transform=transform)

trainset = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last = True)
testset = DataLoader(test_set)

model = ConvNet() 
if GPUbool:
    model.to(device)

lossfun = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate, momentum = 0.9)

trainLoss = torch.zeros(numepochs) 
testLoss = torch.zeros(numepochs) 
trainAcc = torch.zeros(numepochs) 
testAcc = torch.zeros(numepochs)

for epochi in range(numepochs):
    
    n=0
    for X,y in trainset :
        if GPUbool:
            X = X.to(device)
            y = y.to(device)
        yHat = model(X)
        loss = lossfun(yHat,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    batchAcc=[]
    batchLoss= []
    
    for X,y in testset:
        if GPUbool:
            X = X.to(device)
            y = y.to(device)
        
        with torch.no_grad():
            yHat = model(X)
            
            loss = lossfun(yHat,y)
        batchLoss.append(loss.item())
        batchAcc.append( torch.mean((torch.argmax(yHat,axis=1)== y ).float()).item())
        
    testLoss[epochi] = np.mean(batchLoss)
    testAcc[epochi] = 100*np.mean(batchAcc)
    current_time = datetime.now()
    print("Current time:", current_time)
    print(f'Finished epoch {epochi+1}/{numepochs}. Test accuracy = {testAcc[epochi]:.2f}%')