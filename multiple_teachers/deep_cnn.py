# coding: utf-8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import numpy as np
import tensorflow as tf
import time

from differential_privacy.multiple_teachers import utils

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('dropout_seed', 123, """seed for dropout.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Nb of images in a batch.""")
tf.app.flags.DEFINE_integer('epochs_per_decay', 350, """Nb epochs per decay""")
tf.app.flags.DEFINE_integer('learning_rate', 5, """100 * learning rate""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """see TF doc""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.用于移动平均的衰减
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.学习速率衰减系数

#tf.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式，初始化的方式有以下几种：
# tf.constant_initializer：常量初始化函数
# tf.random_normal_initializer：正态分布
# tf.truncated_normal_initializer：截取的正态分布
# tf.random_uniform_initializer：均匀分布
# tf.zeros_initializer：全部是0
# tf.ones_initializer：全是1

# tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer) #tf.get_variable 用来获取或创建一个变量
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.  #帮助创建一个权值衰减的初始变量

  Note that the Variable is initialized with a truncated normal distribution. #注意，变量是用截断的正态分布初始化的
  A weight decay is added only if one is specified.  #只有在指定的情况下才添加一个权重衰减

  Args:
    name: name of the variable #变量的名称
    shape: list of ints #整数列表
    stddev: standard deviation of a truncated Gaussian #正态高斯的标准差
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev)) #tf.truncated_normal_initializer：截取的正态分布
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss') #tf.multiply() 为矩阵点乘 这里指multiply这个
                                       # 函数实现的是元素级别的相乘，也就是两个相乘的数元素各自相乘，而不是矩阵乘法，注意和tf.matmul区别
    tf.add_to_collection('losses', weight_decay)#将value以name的名称存储在收集器(collection)中
  return var


# 前向传播
def inference(images, dropout=False):
  """Build the CNN model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
    dropout: Boolean controlling whether to use dropout or not
  Returns:
    Logits
  """
  if FLAGS.dataset == 'mnist':
    first_conv_shape = [5, 5, 1, 64]
  else:
    first_conv_shape = [5, 5, 3, 64]
    # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
    #图像通道数：标准的数码相机有红、绿、蓝三个通道（Channels），每一种颜色的像素值在0-255之间，构成三个堆叠的二维矩阵；灰度图像则只有一个通道，可以用一个二维矩阵来表示。

  # 声明第一层卷积层的变量并实现前向传播过程
  # conv1 卷积层
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=first_conv_shape,
                                         stddev=1e-4,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME') # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))#tf.constant_initializer(value) 初始化一切所提供的值,
    bias = tf.nn.bias_add(conv, biases) #这个函数的作用的将偏差项biases加到conv  我的理解：对于每一个卷积核都有一个对应的偏置量。
    conv1 = tf.nn.relu(bias, name=scope.name)  # 图片乘以卷积核，并加上偏执量，
    #在神经网络中，我们有很多的非线性函数来作为激活函数，比如连续的平滑非线性函数（sigmoid，tanh和softplus），
    # 连续但不平滑的非线性函数（relu，relu6和relu_x）和随机正则化函数（dropout）。
    #所有的激活函数都是单独应用在每个元素上面的，并且输出张量的维度和输入张量的维度一样。
    #tf.nn.relu(features, name = None)  解释：这个函数的作用是计算激活函数relu，即max(features, 0)。

    if dropout:
      conv1 = tf.nn.dropout(conv1, 0.3, seed=FLAGS.dropout_seed) #防止过拟合


  # pool1 池化层
  # 池化卷积结果（conv2d）池化层采用kernel大小为3*3，步数也为2，周围补0，取最大值。
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME',
                         name='pool1')

  # norm1 归一化层
  norm1 = tf.nn.lrn(pool1,
                    4,
                    bias=1.0,
                    alpha=0.001 / 9.0,
                    beta=0.75,
                    name='norm1')

  #近邻归一化(Local Response Normalization)的归一化方法主要发生在不同的相邻的卷积核（经过ReLu之后）的输出之间，
  # 即输入是发生在不同的经过ReLu之后的 feature map 中
  #LRN归一化主要发生在不同的卷积核的输出之间。

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 128],
                                         stddev=1e-4,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    if dropout:
      conv2 = tf.nn.dropout(conv2, 0.3, seed=FLAGS.dropout_seed)


  # norm2
  norm2 = tf.nn.lrn(conv2,
                    4,
                    bias=1.0,
                    alpha=0.001 / 9.0,
                    beta=0.75,
                    name='norm2')

  # pool2将2X2的像素降为1X1的像素
  pool2 = tf.nn.max_pool(norm2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME',
                         name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    #把所有的东西都移到深度，这样我们就能做一个矩阵乘法
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])#
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[dim, 384],
                                          stddev=0.04,
                                          wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))#因为模型使用的是激活函数Relu，所以需要使用正态分布给参数加一点噪声，来打破完全对称并且避免0 梯度
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    if dropout:
      local3 = tf.nn.dropout(local3, 0.5, seed=FLAGS.dropout_seed)
      # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，
      # 升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要


  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights',
                                          shape=[384, 192],
                                          stddev=0.04,
                                          wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    if dropout:
      local4 = tf.nn.dropout(local4, 0.5, seed=FLAGS.dropout_seed)

  # compute logits
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights',
                                          [192, FLAGS.nb_labels],
                                          stddev=1/192.0,
                                          wd=0.0)
    biases = _variable_on_cpu('biases',
                              [FLAGS.nb_labels],
                              tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return logits


def inference_deeper(images, dropout=False):
  """Build a deeper CNN model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
    dropout: Boolean controlling whether to use dropout or not
  Returns:
    Logits
  """
  if FLAGS.dataset == 'mnist':
    first_conv_shape = [3, 3, 1, 96]
  else:
    first_conv_shape = [3, 3, 3, 96]

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=first_conv_shape,
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 96],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 96],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    if dropout:
      conv3 = tf.nn.dropout(conv3, 0.5, seed=FLAGS.dropout_seed)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 192],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)

  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 192, 192],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 192, 192],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv5, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(bias, name=scope.name)
    if dropout:
      conv6 = tf.nn.dropout(conv6, 0.5, seed=FLAGS.dropout_seed)


  # conv7
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 192, 192],
                                         stddev=1e-4,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv7 = tf.nn.relu(bias, name=scope.name)


  # local1
  with tf.variable_scope('local1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv7, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[dim, 192],
                                          stddev=0.05,
                                          wd=0)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local2
  with tf.variable_scope('local2') as scope:
    weights = _variable_with_weight_decay('weights',
                                          shape=[192, 192],
                                          stddev=0.05,
                                          wd=0)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local2 = tf.nn.relu(tf.matmul(local1, weights) + biases, name=scope.name)
    if dropout:
      local2 = tf.nn.dropout(local2, 0.5, seed=FLAGS.dropout_seed)

  # compute logits
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights',
                                          [192, FLAGS.nb_labels],
                                          stddev=0.05,
                                          wd=0.0)
    biases = _variable_on_cpu('biases',
                              [FLAGS.nb_labels],
                              tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(local2, weights), biases, name=scope.name)

  return logits


def loss_fun(logits, labels):
  """Add L2Loss to all the trainable variables. 将L2Loss添加到所有可训练变量中

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    distillation: if set to True, use probabilities and not class labels to
                  compute softmax loss

  Returns:
    Loss tensor of type float.
  """

  # Calculate the cross entropy between labels and predictions
  labels = tf.cast(labels, tf.int64)#转变类型
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(   #把原来的神经网络输出层的softmax和交叉熵cross_entrop合在一起计算，为了追求速度
      logits=logits, labels=labels, name='cross_entropy_per_example')

  # Calculate the average cross entropy loss across the batch. 计算整批处理的平均交叉熵损失
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')#求平均值，第二参数reduction_indices没有指定的话就在所有的元素中取平均值

#tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
# tf.get_collection：从一个结合中取出全部变量，是一个列表
# tf.add_n：把一个列表的东西都依次加起来

  # Add to TF collection for losses
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def moving_av(total_loss):
  """
  Generates moving average for all losses 滑动平均模型

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss. 定义一个滑动平均的类（class）。初始化时给定了衰减率（0.99）和控制衰减率的变量step。
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  # 定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作时
  # 这个列表中的变量都会被更新。
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  return loss_averages_op


def train_op_fun(total_loss, global_step):
  """Train model.梯度下降算法优化+滑动平均模型+直方图

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables. #创建一个优化器并应用于所有可训练变量。 为所有可训练变量添加 滑动平均模型

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate. 影响学习速度的变量
  nb_ex_per_train_epoch = int(60000 / FLAGS.nb_teachers)

  num_batches_per_epoch = nb_ex_per_train_epoch / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * FLAGS.epochs_per_decay)

  initial_learning_rate = float(FLAGS.learning_rate) / 100.0

  # Decay the learning rate exponentially based on the number of steps. 根据步骤的数量，以指数形式衰减学习速率
  lr = tf.train.exponential_decay(initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.滑动平均模型
  loss_averages_op = moving_av(total_loss)

  # Compute gradients.梯度下降算法优化
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)#作用：创建一个梯度下降优化器对象
    grads = opt.compute_gradients(total_loss)#对于在变量列表（var_list）中的变量计算对于损失函数的梯度,这个函数返回一个（梯度，变量）对的列表，其中梯度就是相对应变量的梯度了。

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)#作用：把梯度“应用”（Apply）到变量上面去。其实就是按照梯度下降的方式加到上面去。

  # Add histograms for trainable variables.为可训练变量添加直方图
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Track the moving averages of all trainable variables.跟踪所有可训练变量的移动平均值
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def _input_placeholder():
  """
  This helper function declares a TF placeholder for the graph input data #这个辅助函数声明了图形输入数据的TF占位符
  :return: TF placeholder for the graph input data #图输入数据的TF占位符
  """
  if FLAGS.dataset == 'mnist':
    image_size = 28
    num_channels = 1
  else:
    image_size = 32
    num_channels = 3

  # Declare data placeholder
  train_node_shape = (FLAGS.batch_size, image_size, image_size, num_channels)
  return tf.placeholder(tf.float32, shape=train_node_shape) #返回一个Tensor可能被用作提供一个值的句柄，但不直接评估。


def train(images, labels, ckpt_path, dropout=False):
  """
  This function contains the loop that actually trains the model.#这个函数包含实际训练模型的循环
  :param images: a numpy array with the input data #一个带有输入数据的numpy数组
  :param labels: a numpy array with the output labels #一个带有输出标签的numpy数组
  :param ckpt_path: a path (including name) where model checkpoints are saved #保存模型检查点的路径(包括名称)
  :param dropout: Boolean, whether to use dropout or not
  :return: True if everything went well
  """

  # Check training data
  assert len(images) == len(labels)
  assert images.dtype == np.float32
  assert labels.dtype == np.int32

  # Set default TF graph 设置默认TF图
  with tf.Graph().as_default():#返回一个上下文管理器，使此图形成为默认图形。
    global_step = tf.Variable(0, trainable=False)# 迭代的计数器

    # Declare data placeholder 申报数据占位符
    train_data_node = _input_placeholder()

    # Create a placeholder to hold labels 创建一个占位符来保存标签
    train_labels_shape = (FLAGS.batch_size,)
    train_labels_node = tf.placeholder(tf.int32, shape=train_labels_shape)#TensorFlow提供了一个占位符操作，必须用数据执行。
    #在训练循环（training loop）的后续步骤中，传入的整个图像和标签数据集会被切片，以符合
    # 每一个操作所设置的batch_size值，占位符操作将会填补以符合这个batch_size值。然后使用feed_dict参数，将数据传入sess.run()函数。

    print("Done Initializing Training Placeholders")#完成初始化培训占位符

    #1.inference() —— 尽可能地构建好图表，满足促使神经网络向前反馈并做出预测的要求。
    #inference()函数会尽可能地构建图表，做到返回包含了预测结果（output prediction）的Tensor。
    #2.loss() —— 往inference图表中添加生成损失（loss）所需要的操作（ops）。
    #3.training() —— 往损失图表中添加计算并应用梯度（gradients）所需的操作。

    # Build a Graph that computes the logits predictions from the placeholder构建一个图形，它计算来自占位符的逻辑预测
    if FLAGS.deeper:
      logits = inference_deeper(train_data_node, dropout=dropout)
    else:
      logits = inference(train_data_node, dropout=dropout) #构建CNN模型

    # Calculate loss计算损失
    loss = loss_fun(logits, train_labels_node)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.构建一个图表，用一批示例来训练模型，并更新模型参数。
    train_op = train_op_fun(loss, global_step)#梯度下降算法优化+滑动平均模型+直方图

    # Create a saver.保存和恢复变量，最简单的保存和恢复模型的方法是使用tf.train.Saver 对象
    saver = tf.train.Saver(tf.global_variables())#tf.global_variables都是获取程序中的变量，返回的值是变量的一个列表

    print("Graph constructed and saver created")#创建的图和保护程序

    # Build an initialization operation to run below.构建一个初始化操作，以在下面运行
    init = tf.global_variables_initializer()

    # Create and init sessions
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) #NOLINT(long-line)
    sess.run(init)

    print("Session ready, beginning training loop")#准备好，开始训练循环

    # Initialize the number of batches
    data_length = len(images)
    nb_batches = math.ceil(data_length / FLAGS.batch_size)

    for step in xrange(FLAGS.max_steps):
      # for debug, save start time
      start_time = time.time()

      # Current batch number
      batch_nb = step % nb_batches

      # Current batch start and end indices
      start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)#计算一个批开始和结束索引

      # Prepare dictionnary to feed the session with
      feed_dict = {train_data_node: images[start:end],
                   train_labels_node: labels[start:end]}

      # Run training step
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      # Compute duration of training step
      duration = time.time() - start_time

      # Sanity check
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      # Echo loss once in a while
      if step % 100 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, ckpt_path, global_step=step)

  return True

#用保存在指定路径中的模型计算softmax激活（概率）作为参数
def softmax_preds(images, ckpt_path, return_logits=False):
  """
  Compute softmax activations (probabilities) with the model saved in the path
  specified as an argument
  :param images: a np array of images
  :param ckpt_path: a TF model checkpoint
  :param logits: if set to True, return logits instead of probabilities
  :return: probabilities (or logits if logits is set to True)
  """
  # Compute nb samples and deduce nb of batches 计算nb样本，并推导出批量nb
  data_length = len(images)
  nb_batches = math.ceil(len(images) / FLAGS.batch_size)

  # Declare data placeholder 申报数据占位符
  train_data_node = _input_placeholder()

  # Build a Graph that computes the logits predictions from the placeholder  建立一个图表，计算出占位符的预测
  if FLAGS.deeper:
    logits = inference_deeper(train_data_node)
  else:
    logits = inference(train_data_node) #构建CNN模型

  if return_logits:
    # We are returning the logits directly (no need to apply softmax)
    output = logits
  else:
    # Add softmax predictions to graph: will return probabilities  #添加softmax预测图:将返回概率
    output = tf.nn.softmax(logits) # tf.nn.softmax函数默认（dim=-1）是对张量最后一维的shape=(p,)向量进行softmax计算，得到一个概率向量。

   #滑动平均模型
  # Restore the moving average version of the learned variables for eval. 为eval恢复学习变量的移动平均版本。
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)  #  定义一个滑动平均的类（class）。初始化时给定了衰减率（0.99）和控制衰减率的变量step。
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore) # 在所有可训练的变量上使用滑动平均

  # Will hold the result
  preds = np.zeros((data_length, FLAGS.nb_labels), dtype=np.float32) #返回给定形状和类型的新数组，填充零。

  # Create TF session
  with tf.Session() as sess:#建立会话
    # Restore TF session from checkpoint file
    saver.restore(sess, ckpt_path) #测试阶段使用saver.restore()方法恢复变量

    # Parse data by batch解析数据批处理
    for batch_nb in xrange(0, int(nb_batches+1)):
      # Compute batch start and end indices 计算批量开始和结束索引
      start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)

      # Prepare feed dictionary
      feed_dict = {train_data_node: images[start:end]}

      # Run session ([0] because run returns a batch with len 1st dim == 1)
      preds[start:end, :] = sess.run([output], feed_dict=feed_dict)[0]

  # Reset graph to allow multiple calls
  tf.reset_default_graph()#清除默认图的堆栈，并设置全局图为默认图

  return preds


