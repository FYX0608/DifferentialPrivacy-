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

import numpy as np


def labels_from_probs(probs):
  """
  Helper function: computes argmax along last dimension of array to obtain
  labels (max prob or max logit value) #Helper函数：根据数组的最后一维计算argmax来获得标签（最大概率或最大logit值）
  :param probs: numpy array where probabilities or logits are on last dimension  #numpy数组，其中概率或logits在最后维度上
  :return: array with same shape as input besides last dimension with shape 1
          now containing the labels#数组的形状与输入相同，除了形状1的最后一个维数现在包含标签
  """
  # Compute last axis index计算最后轴指数
  last_axis = len(np.shape(probs)) - 1

  # Label is argmax over last dimension标签是最后一个维度的argmax
  labels = np.argmax(probs, axis=last_axis)#numpy.argmax返回沿轴的最大值的索引。我的理解：每个teacher_id,、sample id 概率最大所对应的第多少个类，即是标签

  # Return as np.int32
  return np.asarray(labels, dtype=np.int32)#numpy.asarray将输入转换为数组。

#这种聚合机制采用了对相同输入进行推理得到的几个模型的softmax / logit输出，并计算候选类的
# 投票的最高 - 噪声 - 最大值，为每个样本选择一个标签：它将拉普拉斯噪声添加到标签计数中，并返回最频繁 标签  我的理解：每个sample id 票数最多的那个类作为标签
def noisy_max(logits, lap_scale, return_clean_votes=False):
  """
  This aggregation mechanism takes the softmax/logit output of several models
  resulting from inference on identical inputs and computes the noisy-max of
  the votes for candidate classes to select a label for each sample: it
  adds Laplacian noise to label counts and returns the most frequent label.
  :param logits: logits or probabilities for each sample #每个样本的逻辑或概率，没经过处理的logits：3d数组(教师id、样本id、每个类的概率)
  :param lap_scale: scale of the Laplacian noise to be added to counts#拉普拉斯噪声的规模将被添加到计数
  :param return_clean_votes: if set to True, also returns clean votes (without
                      Laplacian noise). This can be used to perform the
                      privacy analysis of this aggregation mechanism.#如果设置为True，也将返回干净的选票（没有拉普拉斯噪音）。 这可以用来执行这个聚合机制的隐私分析。
  :return: pair of result and (if clean_votes is set to True) the clean counts
           for each class per sample and the the original labels produced by
           the teachers.#结果对（如果clean_votes设置为True），每个样本的每个类的干净计数和教师产生的原始标签
  """

  # Compute labels from logits/probs and reshape array properly#从logits / probs计算标签并正确地重新整形数组
  labels = labels_from_probs(logits) #根据数组的最后一维计算argmax来获得标签（最大概率或最大logit值）
  labels_shape = np.shape(labels)
  labels = labels.reshape((labels_shape[0], labels_shape[1]))#给数组赋予新的形状而不改变其数据；labels_shape[0]表示teacher_id,、labels_shape[1]表示sample id

  # Initialize array to hold final labels初始化数组以保存最终的标签
  result = np.zeros(int(labels_shape[1]))#numpy.zeros返回给定形状和类型的新数组，用零填充。

  if return_clean_votes:#初始化数组，为每个sample保持干净的选票
    # Initialize array to hold clean votes for each sample
    clean_votes = np.zeros((int(labels_shape[1]), 10))

  # Parse each sample 解析每个sample样本 寻找票数最多的那个类
  for i in xrange(int(labels_shape[1])):
    # Count number of votes assigned to each class  计算分配给每个类的投票数，每个类出现的次数
    label_counts = np.bincount(labels[:, i], minlength=10) #np.bincount()：计算非负整数数组中每个值的出现次数。我的理解：某个样本sample i上每个类的投票数
                                                           #>>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]) array([1, 3, 1, 1, 0, 0, 0, 1])
    if return_clean_votes:
      # Store vote counts for export存储投票计数出口
      clean_votes[i] = label_counts

    # Cast in float32 to prepare before addition of Laplacian noise在加入拉普拉斯噪声之前，先用float32进行准备
    label_counts = np.asarray(label_counts, dtype=np.float32)#numpy.asarray将输入转换为数组。

    # Sample independent Laplacian noise for each class每个类都有独立的Laplacian噪音
    for item in xrange(10):
      label_counts[item] += np.random.laplace(loc=0.0, scale=float(lap_scale)) #numpy.random.laplace从拉普拉斯或指定位置（或平均值）和比例（衰减）的双指数分布绘制样本。


    # Result is the most frequent label结果是最频繁的标签
    result[i] = np.argmax(label_counts)#返回沿轴的最大值的索引  我的理解：某个样本sample i上票数最多的那个类序号

  # Cast labels to np.int32 for compatibility with deep_cnn.py feed dictionaries
  result = np.asarray(result, dtype=np.int32)#为了与deep_cnn.py字典兼容，将标签转换为np.int32

  if return_clean_votes:
    # Returns several array, which are later saved:
    # result: labels obtained from the noisy aggregation
    # clean_votes: the number of teacher votes assigned to each sample and class
    # labels: the labels assigned by teachers (before the noisy aggregation)
    return result, clean_votes, labels
  else:
    # Only return labels resulting from noisy aggregation
    return result

#这种聚合机制采用了几个模型的softmax / logit输出，这些模型是通过对相同输入进行推理得出的，
# 并计算出最频繁的标签。 这是确定性的（没有像上面的noisy_max（）那样的噪声注入
def aggregation_most_frequent(logits):
  """
  This aggregation mechanism takes the softmax/logit output of several models
  resulting from inference on identical inputs and computes the most frequent
  label. It is deterministic (no noise injection like noisy_max() above.
  :param logits: logits or probabilities for each sample
  :return:
  """
  # Compute labels from logits/probs and reshape array properly从logits / probs计算标签并正确地重新整形数组
  labels = labels_from_probs(logits)
  labels_shape = np.shape(labels)
  labels = labels.reshape((labels_shape[0], labels_shape[1]))

  # Initialize array to hold final labels
  result = np.zeros(int(labels_shape[1]))

  # Parse each sample
  for i in xrange(int(labels_shape[1])):
    # Count number of votes assigned to each class
    label_counts = np.bincount(labels[:, i], minlength=10)#np.bincount()：计算非负整数数组中每个值的出现次数。

    label_counts = np.asarray(label_counts, dtype=np.int32)

    # Result is the most frequent label
    result[i] = np.argmax(label_counts)

  return np.asarray(result, dtype=np.int32)


