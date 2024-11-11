import torch
import math
from tqdm.notebook import tqdm_notebook as tqdm
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import time
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class MFS_Missing_Dataset(Dataset):

    def __init__(self, dataset, num, M):
        M_H = np.loadtxt(str(dataset) + '_H_200.csv', delimiter=",", skiprows=0)
        M_L = np.loadtxt(str(dataset) + '_L_200.csv', delimiter=",", skiprows=0)
        m = len(M_H)
        n = len(M_H[0, :])

        self.max = np.max(M_H*M)
        self.min = np.min(M_H*M)
        M_H = (M_H - self.min) / (self.max - self.min)
        M_L = (M_L - np.min(M_L)) / (np.max(M_L) - np.min(M_L))

        Data_H = np.zeros((num + 1, m * n))
        Data_L = np.zeros((num + 1, m * n))
        Missing = np.zeros((num + 1, m * n))
        Data_H[0, :] = M_H.reshape(1, m * n)
        Data_L[0, :] = M_L.reshape(1, m * n)
        Missing[0, :] = M.reshape(1, m * n)

        M1 = np.hstack((M_H, M_L, M))
        np.random.seed(1)
        for i in range(int(num / 2)):
            np.random.shuffle(M1)
            Data_H[i + 1, :] = M1[:, :n].reshape(1, m * n)
            Data_L[i + 1, :] = M1[:, n:n * 2].reshape(1, m * n)
            Missing[i + 1, :] = M1[:, n * 2:].reshape(1, m * n)
        M2 = np.hstack((M_H.T, M_L.T, M.T))
        for i in range(int(num / 2), num):
            np.random.shuffle(M2)
            Data_H[i + 1, :] = M2[:, :m].T.reshape(1, m * n)
            Data_L[i + 1, :] = M2[:, m:m * 2].T.reshape(1, m * n)
            Missing[i + 1, :] = M2[:, m * 2:].T.reshape(1, m * n)
        self.HF = torch.tensor(Data_H).view(num+1,m,n)
        self.LF = torch.tensor(Data_L).view(num+1, m, n)
        self.M = torch.tensor(Missing).view(num+1, m, n)
        self.num = num


    def __getitem__(self, index):
        HF = self.HF[index]
        LF = self.LF[index]
        M = self.M[index]
        return HF, LF, M

    def __len__(self):
        return self.num+1

    def maxmin(self):
        return self.max, self.min


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU())
            return layers

        self.model = nn.Sequential(
            *block(8*Config.Data_area, 256, normalize=False),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

    def forward(self, x):
        x = x.view(x.size()[0], *Config.Data_size)
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        x = self.model(output)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU())
            return layers

        self.model = nn.Sequential(
            *block(Config.Data_area, 128, normalize=False),
            *block(128, 256),
            *block(256, 128),
            nn.Linear(128, Config.Data_area),
            nn.Sigmoid()
        )

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = x.view(x.size()[0], *Config.Data_size)
        output = self.cnn1(x)
        return output

class Gan():
    def __init__(self, dataset, train_batch_size, train_number_epochs, Data_size, alpha, beta, hnum):
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.train_number_epochs = train_number_epochs
        self.Data_size = Data_size
        self.alpha = alpha
        self.beta = beta
        self.hnum = hnum
        if Config.use_gpu:
            self.generator = Generator().cuda()
            self.discriminator = Discriminator().cuda()
        else:
            self.generator = Generator()
            self.discriminator = Discriminator()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=Config.lr_G)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=Config.lr_D)

    def discriminator_loss(self,LF_X, real_X, M):
        # Generator
        G_sample = self.generator(LF_X)

        # Discriminator
        real_label = Variable(torch.ones(LF_X.size(0), 1)).cuda()
        fake_label = Variable(torch.zeros(LF_X.size(0), 1)).cuda()
        D_prob1 = self.discriminator(real_X * M)
        D_prob2 = self.discriminator(G_sample * M)

        # %% Loss
        criterion = torch.nn.BCELoss()
        D_loss1 = criterion(D_prob1, real_label)
        D_loss2 = criterion(D_prob2, fake_label)
        D_loss = D_loss1 + D_loss2
        return D_loss

    def generator_loss(self, LF_X, real_X, M):
        # %% Structure
        # Generator
        G_sample = self.generator(LF_X)

        # Combine with original data
        Hat_New_X = real_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = self.discriminator(G_sample * M)
        # %% Loss
        real_label = Variable(torch.ones(LF_X.size(0), 1)).cuda()
        criterion = torch.nn.BCELoss()
        G_loss1 = criterion(D_prob, real_label)
        similarity1 = pearsonr(Hat_New_X.view(LF_X.size()[0], -1), LF_X.view(LF_X.size()[0], -1))[:, 0]
        similarity2 = pearsonr((Hat_New_X * M).view(LF_X.size()[0], -1), (LF_X * M).view(LF_X.size()[0], -1))[:, 0]
        G_loss2 = torch.abs(torch.mean(similarity1 - similarity2))

        MSE_train_loss = torch.mean((M * real_X - M * G_sample) ** 2) / torch.mean(M)

        G_loss = G_loss1 + self.alpha * MSE_train_loss + self.beta * G_loss2

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * real_X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def train(self):
        train_dataloader = DataLoader(self.dataset, batch_size=self.train_batch_size, shuffle=False)
        for epoch in range(0, self.train_number_epochs):
            for i, data in enumerate(train_dataloader, 0):
                X_H, X_L, M = data
                if Config.use_gpu is True:
                    X_H = Variable(X_H.view(X_H.size()[0], *self.Data_size)).cuda().to(torch.float32)
                    X_L = Variable(X_L.view(X_H.size()[0], *self.Data_size)).cuda().to(torch.float32)
                    M = Variable(M.view(X_H.size()[0], *self.Data_size)).cuda().to(torch.float32)
                else:
                    pass

                self.optimizer_D.zero_grad()
                D_loss_curr = self.discriminator_loss(LF_X=X_L, real_X=X_H, M=M)
                D_loss_curr.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.generator_loss(LF_X=X_L, real_X=X_H, M=M)
                G_loss_curr.backward()
                self.optimizer_G.step()
                if (i + 1) % 10 == 0:
                    print('Epoch: {}'.format(epoch), end='\t')
                    print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())), end='\t')
                    print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())), end='\t')
                    print('G_loss: {:.4}'.format(np.sqrt(G_loss_curr.item())), end='\t')
                    print('D_loss: {:.4}'.format(np.sqrt(D_loss_curr.item())))

    def test(self):
        test_dataloader = DataLoader(self.dataset, batch_size=1)
        test_X_H, test_X_L, test_M = next(iter(test_dataloader))
        max, min = self.dataset.maxmin()
        test_X_L2 = Variable(test_X_L.view(test_X_L.size()[0], *self.Data_size)).cuda().to(torch.float32)
        get_X_H = self.generator(test_X_L2)
        get_X_H = get_X_H.view(*self.Data_size).cpu()
        imputed_X_H = test_M * test_X_H + (1 - test_M) * get_X_H
        imputed_X_H = imputed_X_H * (max - min) + min
        return imputed_X_H


class Config():
    num = 10000
    alpha = 200
    train_batch_size = 256
    train_number_epochs = 100
    Data_size = (1, 20, 10)
    Data_area = 200
    beta = 100
    lr_G = 0.0001
    lr_D = 0.005
    use_gpu = True  # set it to True to use GPU and False to use CPU

if __name__ == '__main__':
    if Config.use_gpu:
        torch.cuda.set_device(0)

    for name in ['beale']:
        for hnum in range(3,31):
            M_all = np.loadtxt('.\M\hnum='+ str(round(hnum,3)) + '_H_200.csv', delimiter=",", skiprows=0)
            Y = np.zeros((Config.Data_area, 10))
            for l in range(10):
                M = M_all[:,l].reshape(20,10)

                dataset = MFS_Missing_Dataset(name, Config.num, M)
                GAN = Gan(dataset, Config.train_batch_size, Config.train_number_epochs, Config.Data_size, Config.alpha, Config.beta, hnum)
                t1=time.time()
                GAN.train()
                t2=time.time()
                y = GAN.test()
                t3=time.time()
                # print('训练时间',t2-t1)
                # print('测试时间', t3 - t2)
                Y[:, l] = y.detach().numpy().reshape(-1,)
            np.savetxt('./GAN-MMC/'+name+'hnum='+str(round(hnum,3))+'_200.csv', Y, delimiter=',')







