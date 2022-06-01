import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
# --- Training dataset --- #
import torch as th
import cv2
import math
import random
seed = np.random.RandomState(112311)

class ValData(data.Dataset):
    def __init__(self, dataset_path, crop_size=[256,256]):
        super().__init__()
        # train_list = train_data_dir + train_filename
        # with open(train_list) as f:
        #     contents = f.readlines()
        #     input_names = [i.strip() for i in contents]
        #     gt_names = [i.strip().replace('input','gt') for i in input_names]
        # self.train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ECCV_2022/diffusion_ema_rain_imagenet/rain_sub1/'
        # self.train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ICIP_Turbulence_files/Tubfaces89/300M/tubimages/'
        # self.train_data_dir = "/media/malsha/47a8802b-e0b7-47a8-8a4d-1649cc3ad408/sar_optical/optical/"
    
        
        self.noisy_path = os.path.join(dataset_path, 'noisy')
        # self.noisy_path = dataset_path
        # self.clean_path = dataset_path
        self.clean_path = os.path.join(dataset_path, 'clean')
        self.images_list = os.listdir(self.noisy_path)

        
        self.crop_size = crop_size

    def __len__(self):
        return len(os.listdir(self.noisy_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        noisy_im = cv2.imread(os.path.join(self.noisy_path, image_filename))
        clean_im = cv2.imread(os.path.join(self.clean_path, image_filename))

        arr1=np.array(clean_im)
        arr2=np.array(noisy_im)
        arr3 = arr1+ 1e-9
        arr3 = np.divide(arr2,arr3)
        

        arr1 = cv2.resize(arr1, (256,256), interpolation=cv2.INTER_LINEAR)
        arr2= cv2.resize(arr2, (256,256), interpolation=cv2.INTER_LINEAR)
        arr3= cv2.resize(arr3, (256,256), interpolation=cv2.INTER_LINEAR)

        ## for grayscale images
        # arr1 = arr1[..., np.newaxis]
        # arr2 = arr2[..., np.newaxis]
        # arr3 = arr3[..., np.newaxis]

        # arr3 = np.square(arr3)

        # # for log data
        # arr1 = (arr1.astype(np.float32) + 1 )/256.0
        # arr2 = (arr2.astype(np.float32) + 1 )/256.0
        # arr1 = np.log(np.absolute(arr1))
        # arr2 = np.log(np.absolute(arr2))
        # # arr1 = arr1.astype(np.float32) / (0.5*np.log(256.0)) - 1
        # # arr2 = arr2.astype(np.float32) / (0.5*np.log(256.0)) - 1
        # arr1 = 2*(arr1.astype(np.float32) + np.log(256.0))/ np.log(256.0) - 1
        # arr2 = 2*(arr2.astype(np.float32) + np.log(256.0))/ np.log(256.0) - 1


        # ## correct normalization for log

        # arr1 = (arr1.astype(np.float32))/255.0
        # arr2 = (arr2.astype(np.float32))/255.0
        # arr1 = arr1*(math.exp(1)-math.exp(-1)) + math.exp(-1)
        # arr2 = arr2*(math.exp(1)-math.exp(-1)) + math.exp(-1)
        # arr1 = np.log(arr1)
        # arr2 = np.log(arr2)
        # arr1 = arr1.astype(np.float32)
        # arr2 = arr2.astype(np.float32)


        arr1 = arr1.astype(np.float32) / 127.5 - 1
        arr2 = arr2.astype(np.float32) / 127.5 - 1
        # arr3 = arr3.astype(np.float32) / 127.5 - 1
        # arr3 = arr3.astype(np.float32)

        arr2 = np.transpose(arr2, [2, 0, 1])
        arr1 = np.transpose(arr1, [2, 0, 1])
        arr3 = np.transpose(arr3, [2, 0, 1])

        # return arr3, {'SR': arr2, 'HR': arr1 , 'Index': image_filename}
        return arr1, {'SR': arr2, 'HR': arr1 , 'Index': image_filename}
        # return arr2, {'SR': arr2, 'HR': arr2 , 'Index': image_filename}

        # return arr1, {'noise': arr2, 'Index': image_filename}


class ValDataNew(data.Dataset):
    def __init__(self, dataset_path, crop_size=[256,256]):
        super().__init__()
        # train_list = train_data_dir + train_filename
        # with open(train_list) as f:
        #     contents = f.readlines()
        #     input_names = [i.strip() for i in contents]
        #     gt_names = [i.strip().replace('input','gt') for i in input_names]
        # self.train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ECCV_2022/diffusion_ema_rain_imagenet/rain_sub1/'
        # self.train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ICIP_Turbulence_files/Tubfaces89/300M/tubimages/'
        # self.train_data_dir = "/media/malsha/47a8802b-e0b7-47a8-8a4d-1649cc3ad408/sar_optical/optical/"
    
        
        # self.noisy_path = os.path.join(dataset_path, 'noisy')
        self.noisy_path = dataset_path
        self.clean_path = dataset_path
        # self.clean_path = os.path.join(dataset_path, 'clean')
        self.images_list = os.listdir(self.noisy_path)

        
        self.crop_size = crop_size

    def __len__(self):
        return len(os.listdir(self.noisy_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        pil_image = cv2.imread(os.path.join(self.noisy_path, image_filename))      ## Clean image
        
        pil_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
        pil_image = np.repeat(pil_image[:,:,np.newaxis],3, axis=2)
        # print(pil_image.shape)
        

        im1 = ((np.float32(pil_image)+1.0)/256.0)**2
        gamma_noise = seed.gamma(size=im1.shape, shape=1.0, scale=1.0).astype(im1.dtype)
        syn_sar = np.sqrt(im1 * gamma_noise)
        pil_image1 = syn_sar * 256-1   ## Noisy image

        
        
        arr1=np.array(pil_image)
        arr2=np.array(pil_image1)
        
        

        arr1 = cv2.resize(arr1, (256,256), interpolation=cv2.INTER_LINEAR)
        arr2= cv2.resize(arr2, (256,256), interpolation=cv2.INTER_LINEAR)
       


        arr1 = arr1.astype(np.float32) / 127.5 - 1
        arr2 = arr2.astype(np.float32) / 127.5 - 1
        

        arr2 = np.transpose(arr2, [2, 0, 1])
        arr1 = np.transpose(arr1, [2, 0, 1])
        

       
        return arr1, {'SR': arr2, 'HR': arr1 , 'Index': image_filename}
        


class ValDataNewReal(data.Dataset):
    def __init__(self, dataset_path, crop_size=[256,256]):
        super().__init__()
        # train_list = train_data_dir + train_filename
        # with open(train_list) as f:
        #     contents = f.readlines()
        #     input_names = [i.strip() for i in contents]
        #     gt_names = [i.strip().replace('input','gt') for i in input_names]
        # self.train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ECCV_2022/diffusion_ema_rain_imagenet/rain_sub1/'
        # self.train_data_dir = '/media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ICIP_Turbulence_files/Tubfaces89/300M/tubimages/'
        # self.train_data_dir = "/media/malsha/47a8802b-e0b7-47a8-8a4d-1649cc3ad408/sar_optical/optical/"
    
        
        # self.noisy_path = os.path.join(dataset_path, 'noisy')
        self.noisy_path = dataset_path
        self.clean_path = dataset_path
        # self.clean_path = os.path.join(dataset_path, 'clean')
        self.images_list = os.listdir(self.noisy_path)

        
        self.crop_size = crop_size

    def __len__(self):
        return len(os.listdir(self.noisy_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        pil_image = cv2.imread(os.path.join(self.noisy_path, image_filename),0)      ## SAR image
        
        # pil_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
        pil_image = np.repeat(pil_image[:,:,np.newaxis],3, axis=2)
        # print(pil_image.shape)
        

        # im1 = ((np.float32(pil_image)+1.0)/256.0)**2
        # gamma_noise = seed.gamma(size=im1.shape, shape=1.0, scale=1.0).astype(im1.dtype)
        # syn_sar = np.sqrt(im1 * gamma_noise)
        # pil_image1 = syn_sar * 256-1   ## Noisy image

        # pil_image = np.repeat(pil_image[:,:,np.newaxis],3, axis=2)
        # pil_image1 = np.repeat(pil_image1[:,:,np.newaxis],3, axis=2)

    


        
        arr1=np.array(pil_image)
        arr2=np.array(pil_image)
        arr3 = arr1 + 1e-9
        # print(arr3.dtype)
        arr3 = np.divide(arr2,arr3)
        

        arr1 = cv2.resize(arr1, (256,256), interpolation=cv2.INTER_LINEAR)
        arr2= cv2.resize(arr2, (256,256), interpolation=cv2.INTER_LINEAR)
        arr3= cv2.resize(arr3, (256,256), interpolation=cv2.INTER_LINEAR)

        ## for grayscale images
        # arr1 = arr1[..., np.newaxis]
        # arr2 = arr2[..., np.newaxis]
        # arr3 = arr3[..., np.newaxis]

        # arr3 = np.square(arr3)

        # # for log data
        # arr1 = (arr1.astype(np.float32) + 1 )/256.0
        # arr2 = (arr2.astype(np.float32) + 1 )/256.0
        # arr1 = np.log(np.absolute(arr1))
        # arr2 = np.log(np.absolute(arr2))
        # # arr1 = arr1.astype(np.float32) / (0.5*np.log(256.0)) - 1
        # # arr2 = arr2.astype(np.float32) / (0.5*np.log(256.0)) - 1
        # arr1 = 2*(arr1.astype(np.float32) + np.log(256.0))/ np.log(256.0) - 1
        # arr2 = 2*(arr2.astype(np.float32) + np.log(256.0))/ np.log(256.0) - 1


        # ## correct normalization for log

        # arr1 = (arr1.astype(np.float32))/255.0
        # arr2 = (arr2.astype(np.float32))/255.0
        # arr1 = arr1*(math.exp(1)-math.exp(-1)) + math.exp(-1)
        # arr2 = arr2*(math.exp(1)-math.exp(-1)) + math.exp(-1)
        # arr1 = np.log(arr1)
        # arr2 = np.log(arr2)
        # arr1 = arr1.astype(np.float32)
        # arr2 = arr2.astype(np.float32)


        arr1 = arr1.astype(np.float32) / 127.5 - 1
        arr2 = arr2.astype(np.float32) / 127.5 - 1
        # arr3 = arr3.astype(np.float32) / 127.5 - 1
        # arr3 = arr3.astype(np.float32)

        arr2 = np.transpose(arr2, [2, 0, 1])
        arr1 = np.transpose(arr1, [2, 0, 1])
        arr3 = np.transpose(arr3, [2, 0, 1])

        # return arr3, {'SR': arr2, 'HR': arr1 , 'Index': image_filename}
        return arr1, {'SR': arr2, 'HR': arr1 , 'Index': image_filename}




