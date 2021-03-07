

from __future__ import print_function
import argparse
import os
# import h5py
import random
import time
import numpy as np
import scipy.io as sio
import torch
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
from DropBlock_attention import DropBlock2D

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--decreasing_lr', default='120,240,420,620,800', help='decreasing strategy')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
opt = parser.parse_args()
#opt.dataroot = 'E:/picture/CIFAR10'
opt.outf = 'model'
opt.cuda = False
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
cudnn.benchmark = False


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

"""
matfn1 = 'E:/matconvnet/matconvnet/Salinas/Salinas_corrected.mat'
data1 = sio.loadmat(matfn1)
Sa = data1['salinas_corrected']
matfn2 = 'E:/matconvnet/matconvnet/Salinas/Salinas_gt.mat'
data2 = sio.loadmat(matfn2)
GroundTruth = data2['salinas_gt']

matfn3 = 'E:/matconvnet/matconvnet/Salinas/PC.mat'
data3 = sio.loadmat(matfn3)
PCData = data3['PCData']

[nRow, nColumn, nBand] = PCData.shape
"""
###PCA变换
def applyPCA(X,numComponents):
    newX=np.reshape(X,(-1,X.shape[2]))
    pca=PCA(n_components=numComponents,whiten=True)
    newX=pca.fit_transform(newX)
    newX=np.reshape(newX,(X.shape[0],X.shape[1],numComponents))
    return newX
####加padding
def padWithZeros(X,margin=2):
    newX=np.zeros((X.shape[0]+2*margin,X.shape[1]+2*margin,X.shape[2]))
    newX[margin:X.shape[0]+margin,margin:X.shape[1]+margin,:] = X
    return newX
def createImageCubes(X,y,windowSize=5,removeZeroLabels=True):
    margin=int((windowSize-1)/2)
    zeroPaddedX=padWithZeros(X,margin=margin)
    patchesData=np.zeros((X.shape[0]*X.shape[1],windowSize,windowSize,X.shape[2]))
    patchesLabels=np.zeros((X.shape[0]*X.shape[1]))
    patchIndex=0
    for r in range(margin,zeroPaddedX.shape[0]-margin):
	    for c in range(margin,zeroPaddedX.shape[1]-margin):
		    patch=zeroPaddedX[r-margin:r+margin+1,c-margin:c+margin+1]
		    patchesData[patchIndex,:,:,:]=patch
		    patchesLabels[patchIndex]=y[r-margin,c-margin]
		    patchIndex=patchIndex+1
    if removeZeroLabels:
        patchesData=patchesData[patchesLabels>0,:,:,:]
        patchesLabels=patchesLabels[patchesLabels>0]
        patchesLabels-=1
    return patchesData,patchesLabels


def splitTrainTestSet(X,y,testRatio,randomState=345):
    X_train,X_test,y_train,y_test=X*(1-testRatio),X*testRatio,y*(1-testRatio),y*testRatio
    return X_train,X_test,y_train,y_test
def flip(data):

    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data



#num_class = 16

matfn1 = '/home/ouc/WJJ/ACGAN_2/data/Indian_pines_corrected.mat'
data1 = sio.loadmat(matfn1)
X = data1['indian_pines_corrected']
matfn2='/home/ouc/WJJ/ACGAN_2/data/Indian_pines_gt.mat'
data2=sio.loadmat(matfn2)
y = data2['indian_pines_gt']
test_ratio=0.90
patch_size=25
pca_components=3
print('Hyperspectral data shape:',X.shape)
print('Label shape:',y.shape)
X_pca=applyPCA(X,numComponents=pca_components)
print('Data shape after PCA :',X_pca.shape)
[nRow, nColumn, nBand] = X_pca.shape
pcdata = flip(X_pca)
groundtruth = flip(y)

num_class = int(np.max(y))


HalfWidth = 32
Wid = 2 * HalfWidth
G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
data = pcdata[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
[row, col] = G.shape

NotZeroMask = np.zeros([row, col])
Wid = 2 * HalfWidth
NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
G = G * NotZeroMask

[Row, Column] = np.nonzero(G)
nSample = np.size(Row)

RandPerm = np.random.permutation(nSample)


"""
X_pca,y=createImageCubes(X_pca,y,windowSize=patch_size)

print('Data cube X shape:',X_pca.shape)
print('Data cube y shape:',y.shape)
Xtrain,Xtest,ytrain,ytest=splitTrainTestSet(X_pca,y,test_ratio)
print('Xtrain shape:',Xtrain.shape)
print('Xtest shape:',Xtest.shape)
"""
nTrain = 2000
nTest = nSample-nTrain
imdb = {}
imdb['datas'] = np.zeros([2 * HalfWidth, 2 * HalfWidth, nBand, nTrain + nTest], dtype=np.float32)
imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)
for iSample in range(nTrain + nTest):
    imdb['datas'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth,
                                     Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth,
                                     :]
    imdb['Labels'][iSample] = G[Row[RandPerm[iSample]],
                                Column[RandPerm[iSample]]].astype(np.int64)
print('Data is OK.')

imdb['Labels'] = imdb['Labels'] - 1

imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
Xtrain=imdb['datas'][:,:,:,:nTrain]
ytrain=imdb['Labels'][:nTrain]
print('Xtrain :',Xtrain.shape)
print('yTrain:',ytrain.shape)
Xtest=imdb['datas']
ytest=imdb['Labels']
print('Xtest :',Xtest.shape)
print('ytest:',ytest.shape)
"""
Xtrain=Xtrain.reshape(-1,patch_size,patch_size,pca_components)
Xtest=Xtest.reshape(-1,patch_size,patch_size,pca_components)
print(' before Xtrain shape:',Xtrain.shape)
print('before Xtest shape:',Xtest.shape)
"""
Xtrain=Xtrain.transpose(3,2,0,1)
Xtest=Xtest.transpose(3,2,0,1)
print('after Xtrain shape:',Xtrain.shape)
print('after Xtest shape:',Xtest.shape)

####Training
class TrainDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset  = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=200, shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=200, shuffle=False, num_workers=0)


def flip(data):

    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data


"""
num_class = int(np.max(GroundTruth))
pcdata = flip(PCData)
groundtruth = flip(GroundTruth)

HalfWidth = 32
Wid = 2 * HalfWidth
G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
data = pcdata[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
[row, col] = G.shape

NotZeroMask = np.zeros([row, col])
Wid = 2 * HalfWidth
NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
G = G * NotZeroMask

[Row, Column] = np.nonzero(G)
nSample = np.size(Row)

RandPerm = np.random.permutation(nSample)
nTrain = 200
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
print('Data is OK.')

imdb['Labels'] = imdb['Labels'] - 1

imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)




train_dataset = dset.matcifar(imdb, train=True, d=3, medicinal=0)

test_dataset = dset.matcifar(imdb, train=False, d=3, medicinal=0)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200,
                                           shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                          shuffle=True, num_workers=0)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

nc = nBand
nb_label = num_class
"""
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

nc = pca_components
nb_label=num_class
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
        #self.Drop2 = nn.Dropout2d(p=0.5)
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
        #self.Drop2 = nn.Dropout2d(p=0.5)
        self.Drop2 = DropBlock2D()
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False)
        #self.disc_linear = nn.Linear(ndf * 2, 1)
        self.aux_linear = nn.Linear(ndf * 2, nb_label+1)
        self.softmax = nn.LogSoftmax(dim=-1)
        #self.sigmoid = nn.Sigmoid()
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
        #s = self.disc_linear(x).squeeze()
        #s = self.sigmoid(s)
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
f_label = torch.LongTensor(opt.batchSize)



real_label = 0.8
fake_label = 0.2

if opt.cuda:
    netD.cuda()
    netG.cuda()
    s_criterion.cuda()
    c_criterion.cuda()
    input, s_label = input.cuda(), s_label.cuda()
    c_label = c_label.cuda()
    f_label = f_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
f_label = Variable(f_label)
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

for epoch in range(1, opt.niter + 1):
    netD.train()
    netG.train()
    right = 0
    if epoch in decreasing_lr:
        optimizerD.param_groups[0]['lr'] *= 0.9
        optimizerG.param_groups[0]['lr'] *= 0.9

    for i,datas in enumerate(train_loader):
        for j in range(2):    ## Update D 10 times for every G epoch
            netD.zero_grad()
            img, label = datas
            batch_size = img.size(0)
            input.resize_(img.size()).copy_(img)
            s_label.resize_(batch_size).fill_(real_label)
            c_label.resize_(batch_size).copy_(label)
            c_output = netD(input)

            #s_errD_real = s_criterion(s_output, s_label)
            c_errD_real = c_criterion(c_output, c_label)
            errD_real =  c_errD_real
            errD_real.backward()
            D_x = c_output.data.mean()

            correct, length = test(c_output, c_label)
            #print('real train finished!')

                # train with fake

            noise.resize_(batch_size, nz, 1, 1)
            noise.normal_(0, 1)
            noise_ = np.random.normal(0, 1, (batch_size, nz, 1, 1))

            noise.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))

            #label = np.random.randint(0, nb_label, batch_size)
            label = np.full(batch_size, nb_label)

            f_label.data.resize_(batch_size).copy_(torch.from_numpy(label))


            fake = netG(noise)
            #s_label.fill_(fake_label)
            c_output = netD(fake.detach())
            #s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake = c_criterion(c_output, f_label)
            errD_fake = c_errD_fake
            errD_fake.backward()
            D_G_z1 = c_output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            #print('fake train finished!')
            ###############
            #  Updata G
            ##############


        netG.zero_grad()
        #s_label.data.fill_(real_label)  # fake labels are real for generator cost
        c_output = netD(fake)
        #s_errG = s_criterion(s_output, s_label)
        c_errG = c_criterion(c_output, c_label)
        errG = c_errG
        errG.backward()
        D_G_z2 = c_output.data.mean()
        optimizerG.step()
        right += correct
        #print('begin spout!')

    if epoch % 5 == 0:
         print('[%d/%d][%d/%d]   D(x): %.4f D(G(z)): %.4f / %.4f=%.4f,  Accuracy: %.4f / %.4f = %.4f'
                % (epoch, opt.niter, i, len(train_loader),
                  D_x, D_G_z1, D_G_z2, D_G_z1 / D_G_z2,
                 right, len(train_loader.dataset), 100. * right / len(train_loader.dataset)))

        #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    if epoch%5==0:
        netD.eval()
        netG.eval()
        test_loss = 0
        right = 0
        all_Label=[]
        all_target=[]
        for data, target in test_loader:
            indx_target = target.clone()
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            #batch_size = data.size(0)
            #noise.resize_(batch_size, nz, 1, 1)
            #noise.normal_(0, 1)
            #noise_ = np.random.normal(0, 1, (batch_size, nz, 1, 1))
            #noise.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))
           
            #fake=netG(noise)
            #output = netD(data)
            #vutils.save_image(data,'%s/real_samples_i_%03d.png' % (opt.outf,epoch))
            #vutils.save_image(fake,'%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))
            output = netD(data)

            test_loss += c_criterion(output, target).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            all_Label.extend(pred)
            all_target.extend(target)
            right += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(test_loader)  # average over number of mini-batch
        acc =float(100. * float(right)) / float(len(test_loader.dataset))
        print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, right, len(test_loader.dataset), acc))
        if acc > best_acc:
            best_acc = acc

        #C = confusion_matrix(target.data.cpu().numpy(), pred.cpu().numpy())
        C = confusion_matrix(all_target, all_Label)
        C = C[:num_class,:num_class]
        np.save('c.npy', C)
        k = kappa(C, np.shape(C)[0])
        AA_ACC = np.diag(C) / np.sum(C, 1)
        AA = np.mean(AA_ACC, 0)
        print('OA= %.5f AA= %.5f k= %.5f' % (acc, AA, k))