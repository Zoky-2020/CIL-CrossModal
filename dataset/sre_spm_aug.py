import torch
import random
import math
import numpy as np 
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T

from .augmix import augmix

def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda',
                mean=(0.5, 0.5, 0.5)):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class mixing_erasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels with different mixing operation.
    normal: original random erasing;
    soft: mixing ori with random pixel;
    self: mixing ori with other_ori_patch;
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 type='normal',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if self.type == 'normal':
                    m = 1.0
                else:
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))
                if self.type == 'self':
                    x2 = random.randint(0, img.size()[1] - h)
                    y2 = random.randint(0, img.size()[2] - w)
                    img[:, x1:x1 + h,
                        y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                   w] + m * img[:, x2:x2 + h,
                                                                y2:y2 + w]
                else:
                    if self.mode == 'const':
                        img[0, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[0, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[0]
                        img[1, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[1, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[1]
                        img[2, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[2, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[2]
                    else:
                        img[:, x1:x1 + h, y1:y1 +
                            w] = (1 - m) * img[:, x1:x1 + h,
                                               y1:y1 + w] + m * _get_pixels(
                                                   self.per_pixel,
                                                   self.rand_color,
                                                   (img.size()[0], h, w),
                                                   dtype=img.dtype,
                                                   device=self.device)
                return img
        return img


class SYSUData_SRE_SPM_Aug(data.Dataset):
    def __init__(self, data_dir,  pre_transform=None, post_transform=None, colorIndex = None, thermalIndex = None, aug_type='sre_spm'):
        
        # data_dir = '../Datasets/SYSU-MM01/'
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        # self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.aug_type = aug_type

        re_prob = 0.5
        mix_cof = [0.5, 1.0]

        self.random_erasing = mixing_erasing(
            probability=re_prob,
            type='soft',
            mixing_coeff = mix_cof)
        self.re_erasing = mixing_erasing(
            probability=re_prob,
            type='self',
            mixing_coeff = mix_cof)

        self.transform_thermal = T.Compose([
            pre_transform,
            post_transform,
        ])
        
        self.pre_transform = pre_transform
        self.post_transform = T.Compose([
            T.ToTensor(),
            post_transform, ])
    

    def __getitem__(self, index):

        img_rgb,  target_rgb = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img_thermal,  target_thermal = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img_rgb_ori = self.pre_transform(img_rgb)

        if self.aug_type == 'sre_spm':
            img_rgb = self.random_erasing(img_rgb_ori)
            img_rgb = self.re_erasing(img_rgb)

        img_rgb = T.ToPILImage()(img_rgb).convert('RGB')
        img_rgb = np.asarray(img_rgb) / 255.
        img_rgb_ori = np.asarray(T.ToPILImage()(img_rgb_ori).convert('RGB'), dtype=np.uint8)

        img_rgb_1 = augmix(img_rgb)
        img_rgb_2 = augmix(img_rgb)

        img_rgb_1 = np.clip(img_rgb_1*255., 0, 255).astype(np.uint8)
        img_rgb_2 = np.clip(img_rgb_2*255., 0, 255).astype(np.uint8)

        
        img_rgb = self.post_transform(img_rgb_ori)
        img_rgb_1 = self.post_transform(img_rgb_1)
        img_rgb_2 = self.post_transform(img_rgb_2)

        img_rgb_tuple = [img_rgb, img_rgb_1, img_rgb_2]

        img_thermal = self.transform_thermal(img_thermal)
        img_thermal_tuple = [img_thermal, img_thermal, img_thermal]

        return img_rgb_tuple, img_thermal_tuple, target_rgb, target_rgb

    def __len__(self):
        return len(self.train_color_label)


def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label


class RegDBData_SRE_SPM_Aug(data.Dataset):
    def __init__(self, data_dir, trial, pre_transform=None, post_transform=None, colorIndex = None, thermalIndex = None, aug_type='sre_spm'):
        # Load training images (path) and labels
        
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.aug_type = aug_type

        re_prob = 0.5
        mix_cof = [0.5, 1.0]

        self.random_erasing = mixing_erasing(
            probability=re_prob,
            type='soft',
            mixing_coeff = mix_cof)
        self.re_erasing = mixing_erasing(
            probability=re_prob,
            type='self',
            mixing_coeff = mix_cof)

        self.transform_thermal = T.Compose([
            pre_transform,
            post_transform,
        ])
        
        self.pre_transform = pre_transform
        self.post_transform = T.Compose([
            T.ToTensor(),
            post_transform, ])


    def __getitem__(self, index):

        img_rgb,  target_rgb = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img_thermal,  target_thermal = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img_rgb_ori = self.pre_transform(img_rgb)

        if self.aug_type == 'sre_spm':
            img_rgb = self.random_erasing(img_rgb_ori)
            img_rgb = self.re_erasing(img_rgb)        

        img_rgb = T.ToPILImage()(img_rgb).convert('RGB')
        img_rgb = np.asarray(img_rgb) / 255.
        img_rgb_ori = np.asarray(T.ToPILImage()(img_rgb_ori).convert('RGB'), dtype=np.uint8)        

        img_rgb_1 = augmix(img_rgb)
        img_rgb_2 = augmix(img_rgb)

        img_rgb_1 = np.clip(img_rgb_1*255., 0, 255).astype(np.uint8)
        img_rgb_2 = np.clip(img_rgb_2*255., 0, 255).astype(np.uint8)

        img_rgb = self.post_transform(img_rgb_ori)
        img_rgb_1 = self.post_transform(img_rgb_1)
        img_rgb_2 = self.post_transform(img_rgb_2)

        img_rgb_tuple = [img_rgb, img_rgb_1, img_rgb_2]

        img_thermal = self.transform_thermal(img_thermal)
        img_thermal_tuple = [img_thermal, img_thermal, img_thermal]

        return img_rgb_tuple, img_thermal_tuple, target_rgb, target_rgb

    def __len__(self):
        return len(self.train_color_label)