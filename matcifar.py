
from __future__ import print_function
from PIL import Image
import os
import torch
import os.path
import errno
import numpy as np
import scipy.io as sio
import torch.utils.data as data
import torchvision.transforms as transforms





class matcifar1(data.Dataset):


    
    def __init__(self, imdb,train, d, medicinal):
        

        
        self.train = train  # training set or test set
        self.imdb=imdb
        self.d=d
        self.x1=np.argwhere(self.imdb['set']==1)
        self.x2=np.argwhere(self.imdb['set']==3)
        self.x3=np.argwhere(self.imdb['set']>0)
        self.x1=self.x1.flatten()
        self.x2=self.x2.flatten()
        self.x3=self.x3.flatten()

        if medicinal==1:
            self.train_data=self.imdb['data'][self.x1,:,:,:]
            self.train_labels=self.imdb['Labels'][self.x1]
            self.test_data=self.imdb['data'][self.x2,:,:,:]
            self.test_labels=self.imdb['Labels'][self.x2]
            
        else:
            
            self.train_data=self.imdb['data'][:,:,:,self.x1]
            self.train_labels=self.imdb['Labels'][self.x1]
            self.test_data=self.imdb['data'][:,:,:,self.x2]
            self.test_labels=self.imdb['Labels'][self.x2]
            self.all_data=self.imdb['data'][:,:,:,self.x3]
            self.all_labels=self.imdb['Labels'][self.x3]

            if self.d==3:
                self.train_data=self.train_data.transpose((3, 2, 0, 1))
                self.test_data=self.test_data.transpose((3, 2, 0, 1))
                self.all_data=self.all_data.transpose((3, 2, 0, 1))
                self.train_data = torch.tensor(self.train_data)

            else:
                self.train_data=self.train_data.transpose((3, 0, 2, 1))
                self.test_data=self.test_data.transpose((3, 0, 2,1))
                
       


    def __getitem__(self, index):
        
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train==1:
            
            img, target = self.train_data[index], self.train_labels[index]
        if self.train==2:
            
            img, target = self.test_data[index], self.test_labels[index]
        if self.train==3:
            img, target = self.all_data[index], self.all_labels[index]





        return img,target

    def __len__(self):
        if self.train==1:
            return len(self.train_data)
        if self.train==2:
            return len(self.test_data)
        if self.train==3:
            return len(self.all_data)


class cloth(data.Dataset):

    def __init__(self, imdb, train, transform=None, target_transform=None):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.transform = transform
        self.target_transform = target_transform
        self.train_data = self.imdb['train_data']
        self.train_labels = self.imdb['train_labels']
        self.test_data = self.imdb['test_data']
        self.test_labels = self.imdb['test_labels']
        self.all_data = self.imdb['all_data']
        self.all_labels = self.imdb['all_labels']



        self.train_data = self.train_data.transpose((0, 3, 1 , 2))
        self.test_data = self.test_data.transpose((0, 3, 1, 2))
        self.all_data = self.all_data.transpose((0, 3, 1, 2))


    def __getitem__(self, index):

        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train==1:

            img, target = self.train_data[index], self.train_labels[index]
        if self.train==2:

            img, target = self.test_data[index], self.test_labels[index]
        if self.train==3:
            img, target = self.all_data[index], self.all_labels[index]


        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train==1:
            return len(self.train_data)
        if self.train==2:
            return len(self.test_data)
        if self.train==3:
            return len(self.all_data)




