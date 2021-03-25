mport numpy as np
import pandas as pd
import pydicom
import os
import warnings
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import tensorflow as tf
from tensorflow.data import Dataset
from scipy.stats import skew, kurtosis
import gzip

fails = []
def return_default_value_if_fails(default_value):

    def decorator(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                fails.append((func, (args, kwargs), e))
                return default_value
        return inner

    return decorator


def import_image(path):
    try:
        path = path.decode('utf-8')
    except:
        try:
            path = path.numpy().decode('utf-8')
        except:
            pass
    print('processing', path)
    filenames = os.listdir(path)
    slices = [pydicom.dcmread(path + "/" + i) for i in filenames]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    image = np.stack([s.pixel_array.astype(float) for s in slices])
    
    return {'image': image, 'metadata': slices[0]}
    

# ### Bounding box

class CropBoundingBox:
    @staticmethod
    def bounding_box(img3d: np.array):
        mid_img = img3d[int(img3d.shape[0] / 2)]
        same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
        same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
        if same_first_col and same_first_row:
            return True
        else:
            return False

    def __call__(self, sample):
        image = sample['image']
        if not self.bounding_box(image):
            return sample

        mid_img = image[int(image.shape[0] / 2)]
        r_min, r_max = None, None
        c_min, c_max = None, None
        for row in range(mid_img.shape[0]):
            if not (mid_img[row, :] == mid_img[0, 0]).all() and r_min is None:
                r_min = row
            if (mid_img[row, :] == mid_img[0, 0]).all() and r_max is None \
                    and r_min is not None:
                r_max = row
                break

        for col in range(mid_img.shape[1]):
            if not (mid_img[:, col] == mid_img[0, 0]).all() and c_min is None:
                c_min = col
            if (mid_img[:, col] == mid_img[0, 0]).all() and c_max is None \
                    and c_min is not None:
                c_max = col
                break

        image = image[:, r_min:r_max, c_min:c_max]
        return {
            'image': image,
            'metadata': sample['metadata']
        }

class ConvertToHU:
    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        img_type = data.ImageType
        is_hu = img_type[0] == 'ORIGINAL' and not (img_type[2] == 'LOCALIZER')
        if not is_hu:
            warnings.warn(f'Patient {data.PatientID} CT Scan not cannot be'
                          f'converted to Hounsfield Units (HU).')

        intercept = data.RescaleIntercept
        slope = data.RescaleSlope
        image = (image * slope + intercept).astype(np.int16)
        return {'image': image, 'metadata': data}

class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, sample):
        image = sample['image']
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return {
            'image': image,
            'metadata': sample['metadata']
        }
    

class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        resize_factor = np.array(self.output_size) / np.array(image.shape)
        image = zoom(image, resize_factor, mode='nearest')
        return {
            'image': image,
            'metadata': sample['metadata']
        }

class Mask:
    def __init__(self, threshold=-400):
        self.threshold = threshold

    def __call__(self, sample):
        image = sample['image']
        for slice_id in range(image.shape[0]):
            m = self.get_morphological_mask(image[slice_id])
            image[slice_id][m == False] = image[slice_id].min()

        return {
            'image': image,
            'metadata': sample['metadata']
        }

    def get_morphological_mask(self, image):
        m = image < self.threshold
        m = clear_border(m)
        m = label(m)
        areas = [r.area for r in regionprops(m)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(m):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        m[coordinates[0], coordinates[1]] = 0
        return m > 0


class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, sample):
        image = sample['image'].astype(np.float)
        image = (image - self.min) / (self.max - self.min)
        return {
            'image': image,
            'metadata': sample['metadata']
        }
    
class ZeroCenter:
    def __init__(self, pre_calculated_mean):
        self.pre_calculated_mean = pre_calculated_mean

    def __call__(self, sample):
        return {
            'image': sample['image'] - self.pre_calculated_mean,
            'metadata': sample['metadata']
        }
    
class LungStatistics:
    @staticmethod
    def get_lung_px(img):
        lp = img[np.where(img > img.min())]
        return len(lp)
    
    @return_default_value_if_fails(np.array([6000., 0.766346, 0.287670,1.243502, 4.158110])[None, ...])
    def __call__(self, sample):
        # get slice with the maximum lung pixel size
        lps = []
        imgs = sample['image']
        for i in range(imgs.shape[0]):
            lps.append(
            self.get_lung_px(imgs[i, : ,:])
            )
        slice_max = np.where(lps == np.max(lps))[0]
        img_max = imgs[slice_max, :, :]
        
        # get lung volume
        slice_thickness = float(np.max(sample['metadata'].SliceThickness))
        pixel_spacing = float(np.max(sample['metadata'].PixelSpacing[0]))
        #return slice_max
        lung_vol = float(np.array(lps)[slice_max] * slice_thickness * pixel_spacing)
        
        
        # get mean, variance, skew and kurtosis
        non_zero_pixels = img_max[img_max > img_max.min()]
        mean = np.mean(non_zero_pixels)
        var = np.var(non_zero_pixels)
        skew_ = skew(non_zero_pixels)
        kurt = kurtosis(non_zero_pixels)
        
        
        return np.array([lung_vol, mean, var, skew_, kurt])[None, ...]


def image_processing_pipeline(path, return_stats_only = False):
    
    smp = import_image(path)
    smp = CropBoundingBox()(smp)
    smp = ConvertToHU()(smp)
    smp = Clip()(smp)
    smp = Resize((40, 256, 256))(smp)
    smp = Mask()(smp)
    smp = Normalize()(smp)
    smp = ZeroCenter(0.029105728564346046)(smp)
    stats = LungStatistics()(smp)
    if return_stats_only:
        return stats
    
    return smp['image'], stats