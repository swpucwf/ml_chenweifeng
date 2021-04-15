"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf

try:
    xrange
except:
    xrange = range

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def preprocess(path, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation

    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    # 读取灰度图
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # Must be normalized
    # 归一化
    image = image / 255.
    label_ = label_ / 255.
    # zoom：类型为float或sequence，沿轴的缩放系数。 如果float，每个轴的缩放是相同的。 如果sequence，zoom应包含每个轴的一个值。
    # output：放置输出的数组，或返回数组的dtype
    # order：样条插值的顺序，默认为3.顺序必须在0-5范围内。
    # prefilter: bool, optional 。参数预滤波器确定输入是否在插值之前使用spline_filter进行预过滤（对于 > 1
    # 的样条插值所必需的）。 如果为False，则假定输入已被过滤。 默认为True。
    input_ = scipy.ndimage.interpolation.zoom(input=label_,zoom=(1. / scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input=input_,zoom=(scale / 1.), prefilter=False)

    return input_, label_


def prepare_data(sess, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset
      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    dataset:
        "Train" or "Test":to choose the data is train or test
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        #  获取数据目录
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        # 获取测试集路径
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    # 返回文件目录
    return data


def make_data(sess, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    要缩小和放大原始图像，首先要做的是在缩放操作时没有剩余。
    我们需要找到高度(和宽度)和比例因子的模。
    然后，从原始图像的高度(和宽度)中减去模。
    即使经过缩放操作，也不会有余数。
    """
    if len(image.shape) == 3:
        # 取整
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def input_setup(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    if config.is_train:

        data = prepare_data(sess, dataset="Train")
    else:
        data = prepare_data(sess, dataset="Test")

    sub_input_sequence = []
    sub_label_sequence = []
    # 计算padding
    padding = abs(config.image_size - config.label_size) / 2  # 6

    if config.is_train:
        for i in xrange(len(data)):
            # 获取原图和低分辨率还原标签
            input_, label_ = preprocess(data[i], config.scale)
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape

            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                    sub_label = label_[x + int(padding):x + int(padding) + config.label_size,
                                y + int(padding):y + int(padding) + config.label_size]  # [21 x 21]

                    # Make channel value
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)
    else:
        input_, label_ = preprocess(data[1], config.scale)
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape

        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            # 保存索引
            nx += 1
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                sub_label = label_[x + int(padding):x + int(padding) + config.label_size,
                            y + int(padding):y + int(padding) + config.label_size]  # [21 x 21]

                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    make_data(sess, arrdata, arrlabel)

    if not config.is_train:
        return nx, ny


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def merge(images, size):
    # 合并图片
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img
