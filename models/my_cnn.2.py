import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(    #(32,32,3)
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1  #padding=(kernel_size-1)/2
            ),                       #(32,32,6)
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2) #(16,16,16)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2)     #(8,8,32)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2)  #(4,4,256)
        )
        #self.fc1=nn.Linear(32*8*8,120)
        # self.fc1 = nn.Linear(256*4*4, 120)
        # self.out=nn.Linear(120,10)



        #Global average pooling
        self.out=nn.Linear(256,10)



    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        # #GAP
        x=F.avg_pool2d(x,kernel_size=(4,4))
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.out(x)

        # x=x.view(x.size(0),-1)
        # x=F.relu(self.fc1(x))
        # x=self.out(x)
        return x






