# Need test, not finish

import tensorflow as tf
import numpy as np
import math
import sys
import os
import tf_util
from pointnet_util import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))


def placeholder_inputs(batch_size, num_point1, num_feat1, num_point2, num_feat2):
    # Modified
    # 3 + num_feat1: xyz + num_feat1
    pc1 = tf.placeholder(tf.float32, shape=(batch_size, num_point1, 3 + num_feat1))
    pc2 = tf.placeholder(tf.float32, shape=(batch_size, num_point2, 3 + num_feat2))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point1, 3))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point1))
    return pc1, pc2, labels_pl, masks_pl


def get_flownet3d_model(pc1_xyz, pc1_feat, pc2_xyz, pc2_feat, is_training, bn_decay=None, reuse=False):
    # Modify
    """ FlowNet3D
        input:
            batch_size = 1
            pc1_xyz: [batch_size, N1, 3]
            pc1_feat: [batch_size, N1, c]
            pc2_xyz: [batch_size, N2, 3]
            pc2_feat: [batch_size, N2, c]
        output:
            net(optical flow): [batch_size, N1, 3]
            end_points(down sampling point, option): dict
    """
    end_points = {}
    # batch_size = pc1.get_shape()[0].value
    # num_point = pc1.get_shape()[1].value    # num_point1

    l0_xyz_f1 = pc1_xyz
    l0_points_f1 = pc1_feat
    l0_xyz_f2 = pc2_xyz
    l0_points_f2 = pc2_feat

    RADIUS1 = 0.5
    RADIUS2 = 1.0
    RADIUS3 = 2.0
    RADIUS4 = 4.0
    with tf.variable_scope('sa1') as scope:
        if reuse:
            scope.reuse_variables()

        # Frame 1, Layer 1
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=8192, radius=RADIUS1, nsample=256, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        end_points['l1_indices_f1'] = l1_indices_f1

        # Frame 1, Layer 2
        l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=2048, radius=RADIUS2, nsample=256, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        end_points['l2_indices_f1'] = l2_indices_f1

        if not reuse:
            scope.reuse_variables()

        # Frame 2, Layer 1
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=8192, radius=RADIUS1, nsample=256, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        # Frame 2, Layer 2
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2, npoint=2048, radius=RADIUS2, nsample=256, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    _, l2_points_f1_new = flow_embedding_module(l2_xyz_f1, l2_xyz_f2, l2_points_f1, l2_points_f2, radius=3.0, nsample=256, mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding', bn=True, pooling='max', knn=True, corr_func='concat')

    # Layer 3
    l3_xyz_f1, l3_points_f1, l3_indices_f1 = pointnet_sa_module(l2_xyz_f1, l2_points_f1_new, npoint=512, radius=RADIUS3, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    end_points['l3_indices_f1'] = l3_indices_f1

    # Layer 4
    l4_xyz_f1, l4_points_f1, l4_indices_f1 = pointnet_sa_module(l3_xyz_f1, l3_points_f1, npoint=256, radius=RADIUS4, nsample=64, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    end_points['l4_indices_f1'] = l4_indices_f1

    # Feature Propagation
    l3_feat_f1 = set_upconv_module(l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1, nsample=4, radius=2.4, mlp=[], mlp2=[256,256], scope='up_sa_layer1', is_training=is_training, bn_decay=bn_decay, knn=True)
    l2_feat_f1 = set_upconv_module(l2_xyz_f1, l3_xyz_f1, tf.concat(axis=-1, values=[l2_points_f1, l2_points_f1_new]), l3_feat_f1, nsample=4, radius=1.2, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer2', is_training=is_training, bn_decay=bn_decay, knn=True)
    l1_feat_f1 = set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_feat_f1, nsample=4, radius=0.6, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer3', is_training=is_training, bn_decay=bn_decay, knn=True)
    l0_feat_f1 = pointnet_fp_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_feat_f1, [256,256], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_feat_f1, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net


if __name__ == '__main__':
    # Not tested
    with tf.Graph().as_default():
        pc1_xyz = tf.zeros((32, 1024, 3))
        pc1_feat = tf.zeros((32, 1024, 4))
        pc2_xyz = tf.zeros((32, 1024, 3))
        pc2_feat = tf.zeros((32, 1024, 4))
        outputs = get_flownet3d_model(pc1_xyz, pc1_feat, pc2_xyz, pc2_feat, tf.constant(True))
        print(outputs)
