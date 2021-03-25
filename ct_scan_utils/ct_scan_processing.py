import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import pydicom
import os
import re
from tqdm import tqdm
import seaborn as sns
from PIL import Image
from IPython.display import Image as show_gif
from scipy import ndimage
tfd = tfp.distributions
tfb = tfp. bijectors
from skimage import morphology
from skimage import measure
from sklearn import cluster
from scipy.stats import skew, kurtosis
from time import time


def create_image(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def masking(img):
    row_size = img.shape[0]
    col_size = img.shape[1]

    # standardize values
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img/std

    max = np.max(img)
    min = np.min(img)
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)]
    mean = np.mean(middle)
    img[img==max] = mean
    img[img==min] = mean

    # differentiate between tissues and air/empty space by kmeans
    kmeans = cluster.KMeans(2)
    kmeans.fit(img)
    centers = sorted(kmeans.cluster_centers_.flatten())
    #apply threshold
    threshold = np.mean(centers)
    thres_img = np.where(img < threshold, 1., 0.)

    # apply erosion and dilation to differentiate the lungs further
    thres_image_erosion = morphology.erosion(thres_img, np.ones([3,3]))
    thres_image_dilation = morphology.dilation(thres_image_erosion, np.ones([8,8]))

    # obtain labels for boundaries
    labels = measure.label(thres_image_dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)

    # determine if bounds are good (i.e. they cover the entirety of the lungs)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    # initialize mask with zeros (which means that this will be black)
    mask = np.zeros([row_size,col_size],dtype=np.int8)

    # add the good labels to the mask
    mask_size = 0
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
        mask_size += np.sum(labels == N)

    mask = morphology.dilation(mask,np.ones([10, 10]))
    final_img = mask * img

    # apply mask to image
    return final_img, mask_size



def image_preprocess(pid_dir, sample = None, img_size = [512, 512]):
    """
    overall preprocessing step for volumetric images.
    pid_dir = directory for corresponding patient id
    sample = number of images to use for volume, will try to sample 2d images at equidistant steps
    """
    pid_dir = pid_dir.numpy().decode("utf-8")
    print("processing", pid_dir)
    filenames = os.listdir(pid_dir)
    slices = [pydicom.dcmread(pid_dir + "/" + i) for i in filenames]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                                slices[1].SliceLocation)

    if sample is None:
        sample = len(slices)
    slices = [slices[j] for i,j in enumerate(range(0, len(slices), len(slices)//sample)) if i < sample]
    pixel_spacing = np.unique(list(slices[0]['PixelSpacing']))

    img = create_image(slices)
    #resize image
    img = tf.transpose(img, perm = [1,2,0])
    img = tf.image.resize(img, img_size).numpy()

    #return new_img
    lung_pixel_size = []
    masked_imgs = []
    for i in range(img.shape[2]):
        masked, lp_size = masking(img[:, :, i])
        lung_pixel_size.append(lp_size)
        masked_imgs.append(masked)

    masked_imgs = tf.transpose(np.stack(masked_imgs), perm = [1,2,0])

    return tf.cast(masked_imgs, tf.float32), slice_thickness[None,...], pixel_spacing, np.array(lung_pixel_size)

def get_volume(img, slice_thickness, pixel_spacing, lung_pixel_size):
    img = img.numpy()
    s = slice_thickness.numpy()
    p = pixel_spacing.numpy()

    # determine the maximum lung pixel size
    max_lung_px = tf.math.reduce_max(lung_pixel_size)
    idx = tf.reshape(tf.where(lung_pixel_size == max_lung_px), [-1])
    volume = tf.cast(max_lung_px,tf.float32) * s * p


    return img[:, :, idx], volume

def calculate_statistics(img):
    img = img.numpy()
    non_zero_idx = np.where(img > 0)
    non_zero_pixels = img[non_zero_idx[0], non_zero_idx[1]]

    # calculate mean
    mean = np.mean(non_zero_pixels)
    # calculate variance
    var = np.var(non_zero_pixels)
    #calculate skew
    skew_ = skew(non_zero_pixels)
    # kurtosis
    kurt = kurtosis(non_zero_pixels)

    return np.array([mean, var, skew_[0], kurt[0]])

if __name__ == "__main__":
    # gather ids and folders for train dataset
    patient_dcm_dict = {}
    root_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
    for dirname, _, filenames in os.walk(root_dir):
        if 'ID' in dirname:
            dirname_ = dirname.replace(root_dir, "")
            patient_dcm_dict[dirname_] = filenames

    train_pids = [root_dir + pid + "/" for pid in list(patient_dcm_dict.keys())]
    dataset = tf.data.Dataset.from_tensor_slices(train_pids)
    dataset = dataset.map(lambda x:
                          tf.py_function(image_preprocess, [x],
                                       [tf.float32, tf.float32, tf.float32, tf.float32]))

    ## calculates the lung volume, and get the pixel array of the max lung volume
    lung_vol_pixel = dataset.map(lambda x, s, p, lp: tf.py_function(get_volume, [x, s, p, lp],
                                                                    [tf.float32, tf.float32]))
    ## calculate needed statistics from masked pixel array
    lung_img_statistics = lung_vol_pixel.map(lambda img, _:
                                             tf.py_function(
                                                 calculate_statistics, [img], [tf.float32]))



    vol_ = [i[1] for i in list(lung_vol_pixel.take(1).as_numpy_iterator())]
    lung_stats = list(lung_img_statistics.take(1).as_numpy_iterator())

    # check if it properly processed the images 
    print(vol_)
    print(lung_stats)

    start = time()
    lung_stats = [tup[1] for  i, tup in enumerate(dataset)]
    end = time()

    print((end - start)/60, 'mins')

    # prepare data frame for lung image statistics
    df_stats = pd.DataFrame(tf.concat(lung_stats, axis = 0).numpy(), 
                columns = ['lung_vol', 'mean', 'var', 'skew', 'kurt'])
    df_stats['PatientId'] = list(patient_dcm_dict.keys())[0:df_stats.shape[0]]

    # sort lung statistics by patient dictionary
    df_stats = df_stats.set_index('PatientId') \
        .loc[list(patient_dcm_dict.keys())] \
        .reset_index()
    print(df_stats.head())

    # save lung statistics to csv in data
    df_stats.to_csv('data/lung_statistics.csv', index = False)




