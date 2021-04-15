from model import SRCNN
from utils import input_setup
import numpy as np
import tensorflow as tf
import pprint
import os

flags = tf.app.flags
# 设置轮次
flags.DEFINE_integer("epoch", 1000, "Number of epoch [1000]")
# 设置批次
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
# 设置image大小
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
# 设置label
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
# 学习率
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
# 图像颜色的尺寸
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
# 对输入图像进行预处理的比例因子大小
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
# 步长
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
# 权重位置
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
# 样本目录
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
# 训练还是测试
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

# 格式化打印
pp = pprint.PrettyPrinter()

def main(_):
    #   打印参数
    pp.pprint(flags.FLAGS.__flags)

    # 没有就新建~
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # Session提供了Operation执行和Tensor求值的环境;
    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_size=FLAGS.image_size,
                      label_size=FLAGS.label_size,
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)

        srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
