from utils import (
    read_data,
    input_setup,
    imsave,
    merge
)
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    xrange
except:
    xrange = range


class SRCNN(object):
    # 模型初始化
    def __init__(self,
                 sess,
                 image_size=33,
                 label_size=21,
                 batch_size=128,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        # 判断灰度图
        self.is_grayscale = (c_dim == 1)

        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.build_model()

    def build_model(self):
        # tf.placeholder(
        # dtype,
        # shape = None,
        # name = None
        # )
        # 定义image,labels 输入形式 N W H C
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        # tf.Variable(initializer, name), 参数initializer是初始化参数，name是可自定义的变量名称，
        # shape为[filter_height, filter_width, in_channel, out_channels]
        # 构建模型参数
        self.weights = {
            'w1': tf.Variable(initial_value=tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(initial_value=tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(initial_value=tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
        }
        # the dim of bias== c_dim
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
        # 构建模型 返回MHWC
        self.pred = self.model()

        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        # 保存和加载模型
        # 如果只想保留最新的4个模型，并希望每2个小时保存一次，
        self.saver = tf.train.Saver(max_to_keep=4,keep_checkpoint_every_n_hours=2)

    def train(self, config):

        if config.is_train:
            # 训练状态
            input_setup(self.sess, config)
        else:
            nx, ny = input_setup(self.sess, config)


        if config.is_train:

            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")

        else:

            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")


        train_data, train_label = read_data(data_dir)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

        tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")

            for ep in xrange(config.epoch):
                # Run by batch images
                batch_idxs = len(train_data) // config.batch_size
                for idx in xrange(0, batch_idxs):
                    batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]

                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss],
                                           feed_dict={self.images: batch_images, self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                              % ((ep + 1), counter, time.time() - start_time, err))

                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)

        else:
            print("Testing...")
            # print(train_data.shape)
            # print(train_label.shape)
            # print("---------")
            result = self.pred.eval({self.images: train_data, self.labels: train_label})
            # print(result.shape)
            result = merge(result, [nx, ny])
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), config.sample_dir)
            image_path = os.path.join(image_path, "test_image.png")
            imsave(result, image_path)

    def model(self):
        # input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_width, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_width 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
        # filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_width, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_width 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
        # strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
        # padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
        # use_cudnn_on_gpu：  bool类型，是否使用cudnn加速，默认为true
        # padding = “SAME”输入和输出大小关系如下：输出大小等于输入大小除以步长向上取整，s是步长大小；
        # padding = “VALID”输入和输出大小关系如下：输出大小等于输入大小减去滤波器大小加上1，最后再除以步长（f为滤波器的大小，s是步长大小）。

        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID',use_cudnn_on_gpu=True) + self.biases['b1'])
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID',use_cudnn_on_gpu=True) + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID',use_cudnn_on_gpu=True) + self.biases['b3']

        return conv3

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        # 目录
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        # 不存在就新建
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 保存
        # 参数
        '''
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True,
        write_state=True,
        strip_default_attrs=False,
        save_debug_info=False)
        '''
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        # 加载模型
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        # 通过checkpoint文件找到模型文件名
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            # 返回path最后的文件名。如果path以／或\结尾，那么就会返回空值。即os.path.split(path)的第二个元素。
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # 加载成功
            return True
        else:
            # 加载失败
            return False
