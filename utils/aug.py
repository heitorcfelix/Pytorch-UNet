import numpy as np
import numpy.matlib as mth

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torch

def get_random_bb_lim(size, min_size=5, min_dim=5, max_dim=50):
    x_min, y_min = np.random.randint(size-max_dim, size=2)
    width, height = np.random.randint(min_dim, max_dim, size=2)
    x_max = x_min + width
    y_max = y_min + height
    
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if y_max - y_min <= min_size:
        y_max = min(y_max + min_size, size)
        y_min = max(y_min - min_size, 0)

    if x_max - x_min <= min_size:
        x_max = min(x_max + min_size, size)
        x_min = max(x_min - min_size, 0)
    
    return x_min, x_max, y_min, y_max
    
    
def get_crop(src, bbox):
    x_min, x_max, y_min, y_max = bbox
    return src[:, y_min:y_max, x_min:x_max]
    

def get_better_bbox_crop(src):
    
    bbox     = get_random_bb_lim(src.shape[0])
    crop     = get_crop(src, bbox)
    return bbox, crop
    

class RandomNoting():
    def __init__(self, p=0.2):
        super().__init__() 
        self.p           = p
        
    def __call__(self, data):
        # Espected image shape [3, height, width]
        
        if self.p >= np.random.random(): 
            image = data['image']
            mask  = data['mask']

            bbox  = get_random_bb_lim(image.shape[1])
            x_min, x_max, y_min, y_max = bbox

            image[:, y_min:y_max, x_min:x_max] = 0
            mask[:, y_min:y_max, x_min:x_max]  = 0
        
            return { 'image': image, 'mask':  mask}
        return data
    
class RandomSaltPeper():
    def __init__(self, p=0.95, snp_p=0.05):
        super().__init__() 
        
        self.p      = p
        self.snp_p  = snp_p

    def __call__(self, data):
        # Espected image shape [3, height, width], max 1
        
        if self.p >= np.random.random(): 
            image = data['image']
            mask  = data['mask']

            bbox  = get_random_bb_lim(image.shape[1])
            x_min, x_max, y_min, y_max = bbox
            noise = np.random.rand(3, y_max-y_min, x_max-x_min)
            p_p = noise < self.snp_p/2
            p_s = np.all(((self.snp_p > noise), (noise  > self.snp_p/2)),axis=0)
            
            image[:, y_min:y_max, x_min:x_max][p_p] = 0
            image[:, y_min:y_max, x_min:x_max][p_s] = 1
            mask[:, y_min:y_max, x_min:x_max]       = 0
            
            return { 'image': image, 'mask':  mask}
        return data
    

class RandomElastic():
    def __init__(self, p=0.9):
        super().__init__()     
        self.p       = p

        
    def elastic_transform(self, image, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        assert len(image.shape)==3

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape[1:]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        image[0] = map_coordinates(image[0], indices, order=1).reshape(shape)
        image[1] = map_coordinates(image[1], indices, order=1).reshape(shape)
        image[2] = map_coordinates(image[2], indices, order=1).reshape(shape)

        return image
        
    def __call__(self, data):
        
        if self.p < np.random.random(): return data
        image = data['image']
        mask  = data['mask']
        
        sigma  = 14
        alpha  = np.random.randint(600, 800)
        
        bbox  = get_random_bb_lim(image.shape[1], max_dim=100, min_dim=20)
        x_min, x_max, y_min, y_max = bbox
        crop  = get_crop(image, bbox)
        elast_def = self.elastic_transform(crop.copy(), alpha, sigma)
        image[:, y_min:y_max, x_min:x_max][elast_def!=0] = elast_def[elast_def!=0] 
        mask[:, y_min:y_max, x_min:x_max]  = 0
        
        return { 'image': image, 'mask':  mask}  
    
    
class ToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        
        image = data['image']
        mask  = data['mask']
        
        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'mask':  torch.from_numpy(mask).type(torch.FloatTensor)
        }