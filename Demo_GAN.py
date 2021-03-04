# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
# import h5py
import random
import time
import numpy as np
import scipy.io as sio
import os
import torch
#import pytz
#import dateutil
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from matcifar import matcifar1
#from DropBlock import DropBlock2D
from DropBlock_attention import DropBlock2D


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, default=4611,help='manual seed')
#parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--decreasing_lr', default='20', help='decreasing strategy')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
opt = parser.parse_args()
opt.outf = 'model'
opt.cuda = True
print(opt)

CRITIC_ITERS = 1
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


matfn1 = '/home/ouc/WJJ/ACGAN_2/data/PaviaU.mat'
data1 = sio.loadmat(matfn1)
Sa = data1['z']
matfn2 = '/home/ouc/WJJ/ACGAN_2/data/PaviaU_gt.mat'
data2 = sio.loadmat(matfn2)
GroundTruth = data2['paviaU_gt']

matfn3 = '/home/ouc/WJJ/ACGAN_2/data/PCU.mat'
data3 = sio.loadmat(matfn3)
PCData = data3['PCData']

[nRow, nColumn, nBand] = PCData.shape



def flip(data):

    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data



num_class = int(np.max(GroundTruth))
pcdata = flip(PCData)
groundtruth = flip(GroundTruth)

HalfWidth = 32
Wid = 2 * HalfWidth
G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
data = pcdata[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
[row, col] = G.shape
print(row,col)
[row1, col1,nband] = data.shape
print(row1,col1,nband)


[Row, Column] = np.nonzero(G)
print("Row2",Row)
print("Colume2",Column)
nSample = np.size(Row)
print("nsample2",nSample)
RandPerm = np.arange(nSample)
print(RandPerm)
nTrain = 1000
nTest = nSample-nTrain


t_begin = time.time()


imdb = {}
imdb['data'] = np.zeros([2 * HalfWidth, 2 * HalfWidth, nBand, nTrain + nTest], dtype=np.float32)
imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)
for iSample in range(nTrain + nTest):
    imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth,
                                     Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth,
                                     :]
    imdb['Labels'][iSample] = G[Row[RandPerm[iSample]],
                                Column[RandPerm[iSample]]].astype(np.int64)
print("label:",imdb['Labels'])
print('Data is OK.')

imdb['Labels'] = imdb['Labels'] - 1

imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
random.shuffle(imdb['set'])





train_dataset = matcifar1(imdb, train=1, d=3, medicinal=0)

all_dataset= matcifar1(imdb, train=3, d=3, medicinal=0)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200,
                                           shuffle=False, num_workers=0)
all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=100,
                                          shuffle=False, num_workers=0)


nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

nc = nBand
nb_label = num_class
print("label",nb_label)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


class netG(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(netG, self).__init__()
        self.ReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)
        self.Drop2 = DropBlock2D()

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ngf)

        self.conv6 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)
        x = self.Drop2(x)
        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)
        x = self.conv6(x)
        output = self.Tanh(x)
        return output


class netD(nn.Module):
    def __init__(self, ndf, nc, nb_label):
        super(netD, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.Drop2 = DropBlock2D()
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False)
        self.aux_linear = nn.Linear(ndf * 2, nb_label+1)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)
        x = self.Drop2(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)
        x = self.Drop2(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 2)
        c = self.aux_linear(x)
        c = self.softmax(c)
        return c



netG = netG(nz, ngf, nc)


if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = netD(ndf, nc, nb_label)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
s_label = torch.FloatTensor(opt.batchSize)
c_label = torch.LongTensor(opt.batchSize)
input_label = torch.LongTensor(opt.batchSize)



real_label = 0.8
fake_label = 0.2

if opt.cuda:
    netD.cuda()
    netG.cuda()
    s_criterion.cuda()
    c_criterion.cuda()
    input, s_label = input.cuda(), s_label.cuda()
    c_label = c_label.cuda()
    input_label=input_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
input_label = Variable(input_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)


optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.02)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.005)


decreasing_lr = list(map(int, opt.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))


def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)


def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / np.sum(dataMat) ** 2
    P0 = float(P0 / np.sum(dataMat) * 1.0)
    cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    return cohens_coefficient



best_acc = 0
trainD_loss=[]
trainG_loss=[]

try:
    for epoch in range(1, opt.niter + 1):
        netD.train()
        netG.train()
        right = 0
        if epoch in decreasing_lr:
            optimizerD.param_groups[0]['lr'] *= 0.9
            optimizerG.param_groups[0]['lr'] *= 0.9

        for i, data in enumerate(train_loader, 1):
            #for j in range(10):    ## Update D 10 times for every G epoch
            for j in range(10):    ## Update D 10 times for every G epoch
                netD.zero_grad()
                img, label = data
                batch_size = img.size(0)
                input.data.resize_(img.size()).copy_(img)
                c_label.data.resize_(batch_size).copy_(label)
                input_label.data.resize_(batch_size).copy_(label)
                c_output = netD(input)
                c_errD_real = c_criterion(c_output, c_label)
                errD_real = c_errD_real
                errD_real.backward()

                D_x = c_output.data.mean()

                correct, length = test(c_output, c_label)

                # train with fake

                noise.data.resize_(batch_size, nz, 1, 1)
                noise.data.normal_(0, 1)
                noise_ = np.random.normal(0, 1, (batch_size, nz, 1, 1))

                noise.data.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))

                label = np.random.randint(0, nb_label+1, batch_size)

                c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))


                fake = netG(noise)
                c_output = netD(fake.detach())
                c_errD_fake = c_criterion(c_output, c_label)
                errD_fake = c_errD_fake
                errD_fake.backward()
                D_G_z1 = c_output.data.mean()
                errD = errD_real + errD_fake
                optimizerD.step()
            
            ###############
            #  Updata G
            ##############


            netG.zero_grad()
            #s_label.data.fill_(real_label)  # fake labels are real for generator cost
            c_output = netD(fake)
            c_errG = c_criterion(c_output, c_label)
            errG = c_errG
            errG.backward()
            
            D_G_z2 = c_output.data.mean()
            optimizerG.step()
            right += correct
 

        if epoch % 10 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f  D(x): %.4f D(G(z)): %.4f / %.4f=%.4f,  Accuracy: %.4f / %.4f = %.4f'
                  % (epoch, opt.niter, i, len(train_loader),
                     errD.item(), D_x, D_G_z1, D_G_z2, D_G_z1 / D_G_z2,
                     right, len(train_loader.dataset), 100. * right / len(train_loader.dataset)))
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

        if epoch % 5 == 0:
        #if epoch % 50 == 0:
            netD.eval()
            netG.eval()
            test_loss = 0
            right = 0

            all_Label=[]
            all_target=[]
            for data, target in all_loader:
                indx_target = target.clone()
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                output = netD(data)


                test_loss += c_criterion(output, target).item()
                pred = output.data.max(1)[1]  
                all_Label.extend(pred)
                all_target.extend(target)
                right += pred.cpu().eq(indx_target).sum()

            
            test_loss = test_loss / len(all_loader)  # average over number of mini-batch
            acc = float(100. * float(right) / float(len(all_loader.dataset)))
            print('\tAll set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, right, len(all_loader.dataset), acc))
            if acc > best_acc:
                #all_Label=np.array(all_Label)
                best_acc = acc


            C = confusion_matrix(all_target, all_Label)
            np.save('c.npy', C)
            k = float(kappa(C, np.shape(C)[0]))
            AA_ACC = np.diag(C) / np.sum(C, 1)
            AA = float(np.mean(AA_ACC, 0))
            print('OA= %.5f AA= %.5f k= %.5f' % (acc, AA, k))


except Exception as e:
    import traceback

    traceback.print_exc()
finally:

    input=input.cpu().detach().numpy()
    input_img = {}
    input_img['input_img'] = input
    sio.savemat(os.path.join(os.getcwd(), 'img/input_img'), input_img)

    input_label=input_label.cpu().detach().numpy()
    real_label = {}
    real_label['real_label'] = input_label
    sio.savemat(os.path.join(os.getcwd(), 'result/real_label'), real_label)
    
    #np.savetxt('G.txt',trainG_loss)
    #np.savetxt('D.txt',trainD_loss)
    
    
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))


