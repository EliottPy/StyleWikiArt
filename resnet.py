import numpy as np 
 
import torch 
 
import torch.nn as nn 
import torch.nn.functional as F 
 
 
import torchvision.transforms
import torchvision.transforms as T 
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
 
import matplotlib.pyplot as plt 
import random
 



transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
batch_size = 32 
 
train_set =  ImageFolder(root="./db/train_resnet18" , transform=transform)
test_set =  ImageFolder(root="./db/valid_resnet18" , transform=transform)
 
trainset = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last = True)
testset = DataLoader(test_set)

 
 
 
resnet = torchvision.models.resnet18(pretrained= True)

GPUbool = torch.cuda.is_available()
if GPUbool:
    resnet.cuda()

 
for p in resnet.parameters():
 
 
    p.requires_grad = False
 
 
resnet.fc = nn.Linear(512,14) 
 
lossfun = nn.CrossEntropyLoss()
optimizer =  torch.optim.SGD(resnet.parameters(),lr = 0.001, momentum = 0.9)

 
numepochs = 5
 
trainLoss = torch.zeros(numepochs)
testLoss = torch.zeros(numepochs)
trainAcc = torch.zeros(numepochs)
testAcc = torch.zeros(numepochs)
 
for epochi in range(numepochs):
    resnet.train()
    if GPUbool:
        resnet.cuda()
 
    for X,y in trainset :
        if GPUbool:
            X = X.cuda()
            y = y.cuda()
        yHat = resnet(X)
        if GPUbool:
            yHat = yHat.cuda()
    

        loss = lossfun(yHat,y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    resnet.eval()
    batchAcc=[]
    batchLoss= []
 
    for X,y in testset:
        if GPUbool:
            X = X.cuda()
            y = y.cuda()
 
        with torch.no_grad():
            yHat = resnet(X)
            if GPUbool:
                yHat = yHat.cuda()
 
 
            loss = lossfun(yHat,y)
        batchLoss.append(loss.item())
        batchAcc.append( torch.mean((torch.argmax(yHat,axis=1)== y ).float()).item())
 
    testLoss[epochi] = np.mean(batchLoss)
    testAcc[epochi] = 100*np.mean(batchAcc)
    print(f'Finished epoch {epochi+1}/{numepochs}. Test accuracy = {testAcc[epochi]:.2f}%')
 
    resnet.eval()
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
 
 
 
for i in range(len(axs.flatten())):
    ax = axs.flatten()[i]
 
    Xt,y = random.choice(testset.dataset)  
 
 
 
    pic = Xt.cpu().numpy().transpose((1,2,0))
    pic = pic - np.min(pic)
    pic = pic/np.max(pic)
 
 
 
    ax.imshow(pic)
 
    Xt = torch.unsqueeze(Xt, 0)
    if GPUbool:
        Xt = Xt.cuda()
    yhat = resnet(Xt)
    if GPUbool:
        yhat = yhat.cuda()
    pred = torch.argmax(yhat,axis=1)
 
 
    label = test_set.classes[pred]
 
    truec = test_set.classes[y]
    title = f'Pred : {label} - true : {truec}'
    titlecolor = 'g' if truec == label else 'r'
 
    ax.text(112, 10, title, ha='center', va='top', fontweight='bold', color='k', backgroundcolor=titlecolor, fontsize=8)
 
    ax.axis('off')
 
plt.tight_layout()
plt.show()