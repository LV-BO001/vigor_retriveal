import os
import numpy as np
import tensorflow as tf


# gt_dist=2
# dist_array=np.random.random((4,3))
# print(dist_array)
# for i in range(dist_array.shape[0]):
#     print(i)
#     print(dist_array[i, :] < gt_dist)
#     print(np.sum(dist_array[i, :] < gt_dist))
#     print("TTT")

#张量扩充
dist_array=tf.zeros((4,2, 5))
print(dist_array)

dist_vector=tf.pad(dist_array,([0,0],[0,0],[0,5]),constant_values=1)
print(dist_vector)
with tf.Session() as sess:
    print(sess.run(dist_vector))


# grd_vector = tf.ones((1,3),dtype=tf.float32)
#
#
# list1=np.random.random((1,3))
# c=tf.concat([grd_vector,list1],0)
# print(c)

#[[1,2,3,4],[1,4,7,9],[4,7,3,8],[3,6,2,9]]
#list1的shape为2×4
# list1=[1,2,3,4,1,4,7,9,11,2,6,1,11,23,45,5,7,3,9,
#        3,5,2,5,3,5,2,6,2,4,3,8,2,9,21,4,2,5,2,5,2,4
# ,6, 2, 4, 3,2,3,4,1,4,8,2,9,21,8,2,9, 2,6,5]

# vec1 = tf.reshape(list1,[4,3,5],name=None)
# print(vec1)
# with tf.Session() as sess:
#     print(sess.run(vec1))
#获取二维数组中第1、2，3行中的第1，3列的数据
# list2=list2[0:3,[0,2]]
#2个4×5的矩阵
# list2 = [[[1,2,4,5,6],[2,4,3,4,5],[2,4,4,2,5],[2,3,4,5,6]],
#          [[3,3,5,3,5],[2,4,3,4,5],[2,4,4,2,5],[1,5,3,7,9]],
#          ]

#list类型转换为tensor张量
# weight1 = tf.reshape(list2,[2,4,5],name=None)
# print(weight1)
# with tf.Session() as sess:
#     print(sess.run(weight1))
# vec2 = tf.einsum('bi, ijd -> bjd', list1, list2)
# vec2 = tf.einsum('bi, ijd -> bjd', vec1, weight1)
# vec2 = np.einsum('bjd, jid -> bid', list1,list2)
# print(vec2)
# with tf.Session() as sess:
#     print(sess.run(vec2))



# w4=[1,2, 4, 5, 6,2, 4, 3, 4, 5,
#   2, 4 ,4 ,2, 5]
# w5= [[3 ,3 ,5, 3, 5],
#   [2 ,4 ,3, 4, 5],
#   [2 ,4, 4, 2 ,5]]
# w6=[[3, 3 ,5, 3, 5],
#   [2 ,4 ,3 ,4, 5],
#   [2 ,4, 3, 4, 5]]
# w7= [[3, 3 ,5, 3, 5],
#   [2 ,4 ,3, 4, 5],
#   [2 ,4 ,3, 4, 5]]
# w4 = tf.reshape(w4,[3,5],name=None)
# w5 = tf.reshape(w5,[3,5],name=None)
# w6 = tf.reshape(w6,[3,5],name=None)
# w7 = tf.reshape(w7,[3,5],name=None)
# w8=w4+2*w5+3*w6+4*w7
# w9=w4+4*w5+7*w6+9*w7
# print("!!!!")
# with tf.Session() as sess:
#     print(sess.run(w8))
# with tf.Session() as sess:
#     print(sess.run(w9))



