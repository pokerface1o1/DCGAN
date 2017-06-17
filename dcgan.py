#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:29:02 2017

@author: Pranjal
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets 
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1)

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])

mnist = datasets.MNIST(root='./data/',
                       train=True,
                       transform=transform,
                       download=True)

data_loader = torch.utils.data.DataLoader(mnist, 
                                          batch_size=100,
                                          shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2),
                nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2),
                nn.BatchNorm2d(32))
        self.fc = nn.Linear(7*7*32, 1)
        
    def forward(self, x):
        out = F.leaky_relu(self.layer1(x), 0.05)
        out = F.leaky_relu(self.layer2(out), 0.05)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.fc(out))
        
        return out

  class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(100, 64*7*7)
        self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2),
                nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2),
                nn.BatchNorm2d(16))
        self.layer3 = nn.ConvTranspose2d(16, 1, kernel_size=5, padding=2)
        
    def forward(self, z):
        out = F.leaky_relu(self.linear(z), 0.05)
        out = out.view(out.size(0), 64, 7, 7)
        out = F.leaky_relu(self.layer1(out), 0.05)
        out = F.leaky_relu(self.layer2(out), 0.05)
        out = F.tanh(self.layer3(out))
        
        return out
    
D = Discriminator()
G = Generator()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003, betas = (0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003, betas = (0.5, 0.999))

D.zero_grad()
G.zero_grad()

for epoch in range(200):
    for i,(images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        images = Variable(images)
        
        output = D(images)
        d_real_loss = torch.mean((output-1)**2)
        real_score = output
        
        z = Variable(2*((torch.randn(batch_size, 100).bernoulli_(0.5))-0.5))
        fake_images = G(z)
        output = D(fake_images)
        d_fake_loss = torch.mean((output)**2)
        fake_score = output
        
        d_loss = d_real_loss + d_fake_loss
        D.zero_grad()
        
        d_loss.backward(retain_variables=True)
        d_optimizer.step()
        
        z = Variable(2*((torch.randn(batch_size, 100).bernoulli_(0.5))-0.5))
        fake_images = G(z)
        output = D(fake_images)
        g_loss = torch.mean((output-1)**2)
        
        D.zero_grad()
        G.zero_grad()
        
        g_loss.backward(retain_variables=True)
        g_optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                  'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f' 
                  %(epoch+1, 200, i+1, 600, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))
            
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), '/Users/Pranjal/Downloads/data/dcgan/fake_images-%d.png' %(epoch+1))

    
