import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

import torchvision
import torchvision.transforms.functional as tf


import sklearn
from sklearn.model_selection import train_test_split


from elpv_dataset.utils import load_dataset
from sklearn.metrics import confusion_matrix

# Loading the dataset and changing it to appropriate dimensions to include a channel.
images,proba,types=load_dataset()
images=images[:,None,:,:]

#Train,CV,Test split
X,X_test,Y,Y_test=train_test_split(images,proba,test_size=0.15,random_state=42)
X_train,X_cv,Y_train,Y_cv=train_test_split(X,Y,test_size=0.15,random_state=42)



# Transforming the continuous values of outputs into 4 classes used for multiclass classification.
Y_train=np.round(Y_train*3).astype(np.int64)
Y_test=np.round(Y_test*3).astype(np.int64)
Y_cv=np.round(Y_cv*3).astype(np.int64)

# converting to tensor from numpy

X_train=torch.from_numpy(X_train)
X_cv=torch.from_numpy(X_cv)
X_test=torch.from_numpy(X_test)
Y_train=torch.from_numpy(Y_train)
Y_cv=torch.from_numpy(Y_cv)
Y_test=torch.from_numpy(Y_test)

X_train=X_train.float()/255.0
X_cv=X_cv.float()/255.0
X_test=X_test.float()/255.0

# Dataset and Dataloaders
train=TensorDataset(X_train,Y_train)
cv=TensorDataset(X_cv,Y_cv)
test=TensorDataset(X_test,Y_test)

train_loader=DataLoader(dataset=train,batch_size=60,shuffle=True)
cv_loader=DataLoader(dataset=cv,batch_size=60,shuffle=True)
test_loader=DataLoader(dataset=test,batch_size=60,shuffle=True)



# CNN architecture
class CNN(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(CNN,self).__init__()

        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=8,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.fc1=nn.Linear(32*37*37,num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=F.relu(self.conv3(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)
        return x

device="cuda" if torch.cuda.is_available() else "cpu"
model=CNN(in_channels=1,num_classes=4).to(device)


# function for calculating accuracy
def check_accuracy(loader,model):
    model.eval()
    correct,total=0,0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device).float(),y.to(device)
            scores=model(x)
            preds=scores.argmax(dim=1)
            correct+= (preds==y).sum().item()
            total+=y.size(0)
    model.train()
    return correct/total



# Training
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
num_epochs=10
for j in range(num_epochs):
    epoch_loss=0.0
    for i,(inputs,labels) in enumerate(train_loader):
        inputs=inputs.to(device).float()
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

    train_acc=check_accuracy(train_loader,model)
    cv_acc=check_accuracy(cv_loader,model)
    print(f"Epoch {j+1}/{num_epochs}, loss= {epoch_loss/len(train_loader):.4f}, train acc:{train_acc:.4f}, cv_acc:{cv_acc:.4f}")
test_acc=check_accuracy(test_loader,model)

print(f"Final test accuracy:{test_acc:.4f}")

model.eval()
all_preds=[]
all_labels=[]
with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device).float()
        scores=model(x)
        preds=scores.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.numpy())
model.train()

all_preds=np.concatenate(all_preds)
all_labels=np.concatenate(all_labels)
print(confusion_matrix(all_labels,all_preds))


#Save the model
torch.save(model.state_dict(),'MultiClassCNN.pth')

loaded_model=CNN(in_channels=1,num_classes=4)

loaded_model.load_state_dict(torch.load('MultiClassCNN.pth'))

print(loaded_model)


  



















