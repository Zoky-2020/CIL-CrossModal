import numpy as np
import torch
from imagecorruptions.corruptions import *
import albumentations as abm
import PIL

def rain(image, severity=1):
    if severity == 1:
        type = 'drizzle'
    elif severity == 2 or severity == 3:
        type = 'heavy'
    elif severity == 4 or severity == 5:
        type = 'torrential'
    blur_value = 2 + severity
    bright_value = -(0.05 + 0.05 * severity)
    rain = abm.Compose([
        abm.augmentations.transforms.RandomRain(rain_type=type,
                                                blur_value=blur_value,
                                                brightness_coefficient=1,
                                                always_apply=True),
        abm.augmentations.transforms.RandomBrightness(
            limit=[bright_value, bright_value], always_apply=True)
    ])
    return rain(image=np.array(image))['image']


class RandomCorruption(object):
    def __init__(self, severity = None):
        self.aug_severity = severity
        self.corruption_type = [gaussian_noise, shot_noise, impulse_noise, 
                defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, 
                fog, brightness, contrast, elastic_transform, pixelate, rain , 
                jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate]

    """
    PIL img
    """
    def __call__(self, img):
        img_aug = img.copy()
        op = np.random.choice(self.corruption_type)
        aug_severity = self.aug_severity
        if aug_severity is None:
            aug_severity = np.random.choice(np.uint8([1,2,3,4,5]))
        img_aug = op(img_aug, aug_severity)
        return PIL.Image.fromarray(np.uint8(img_aug))

