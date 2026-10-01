import math
import torch
import torch.nn as nn
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    else:
        padding = 0
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


class OWM_Net_3(torch.nn.Module):

    def __init__(self, inputsize):
        super(OWM_Net_3, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(256)

        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False)
        
        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        self.fc3 = torch.nn.Linear(1000, 10,  bias=False)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

        return
    

    def forward(self, x, std):
        h_list = []
        x_list = []
            
        kernel3 = get_gaussian_filter(kernel_size=3, sigma=std/1, channels=256).to(device)
        
        x = self.padding(x)
        x_list.append(torch.mean(x, 0, True))
        con1 = self.drop1(self.relu(self.c1(x)))

        con1_p = self.maxpool(con1)

        con1_p = self.padding(con1_p)
        x_list.append(torch.mean(con1_p, 0, True))
        con2 = self.drop1(self.relu(self.c2(con1_p)))

        con2_p = self.maxpool(con2)

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3 = kernel3(con3)

        con3_p = self.maxpool(con3)

        h = con3_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        y = self.fc3(h)
        return y, h_list, x_list

# -------------------------------------------------------------------------------------------------#

