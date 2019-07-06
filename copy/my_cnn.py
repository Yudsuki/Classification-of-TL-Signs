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
            #nn.BatchNorm2d(32,affine=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2) #(16,16,32)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            #nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Dropout(0.5),    #(16,16,64)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            #nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2)    #(8,8,128)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            #nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            nn.Dropout(0.5) #(8,8,256)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(256,512,3,1,1),
            #nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            nn.Dropout(0.5) #(8,8,512)
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(512,1024,3,1,1),
            #nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2)  #(4,4,1024)
        )

        #self.fc1=nn.Linear(32*8*8,120)
        # self.fc1 = nn.Linear(256*4*4, 120)
        # self.out=nn.Linear(120,10)

        #Global average pooling
        self.fc1=nn.Linear(1024,120)
        self.out=nn.Linear(120,10)



    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        #GAP
        x=F.avg_pool2d(x,kernel_size=(4,4))
        print(x.size())
        x = x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x = self.out(x)





        # x=x.view(x.size(0),-1)
        # #x = x.view(-1, 32*8*8)
        # x=F.relu(self.fc1(x))
        # x=self.out(x)
        return x