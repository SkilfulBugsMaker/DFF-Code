# Need test, not finish
import os
import sys

sys.path.append('../utils/')
sys.path.append('../tf_ops/grouping/')

import tensorflow as tf
import numpy as np
import math

import models.tf_util
from models.pointnet_util import *

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(BASE_DIR)
#sys.path.append(os.path.join(ROOT_DIR, '../utils'))
#sys.path.append(os.path.join(ROOT_DIR, '../tf_ops/grouping'))

from tf_grouping import query_ball_point, group_point, knn_point

def get_flowed_feature(
        pc1_xyz, pc1_rgb, pc1_graph_point_xyz, pc1_graph_feat,
        pc2_xyz, pc2_rgb, pc2_graph_point_xyz, pc2_graph_point_idx,
        t_is_training, n_sample=4
):
    # 1 key, 2 non key
    pc_non_key_xyz = tf.expand_dims(pc2_xyz, 0)
    pc_non_key_feat = tf.expand_dims(pc2_rgb, 0)
    pc_key_xyz = tf.expand_dims(pc1_xyz, 0)
    pc_key_feat = tf.expand_dims(pc1_rgb, 0)
    print(pc_non_key_xyz, pc_non_key_feat, pc_key_feat)
    # [1, N_non_key ,3]
    optical_flow_non_key_to_key = get_flownet3d_large_model(
        pc_non_key_xyz,
        pc_non_key_feat,
        pc_key_xyz,
        pc_key_feat,
        is_training=t_is_training)
    print(optical_flow_non_key_to_key)
    # down-sampling optical flow
    # [1, N_non_key_graph, 1 ,3]
    optical_flow_non_key_to_key_graph_points = tf.gather(
        optical_flow_non_key_to_key, pc2_graph_point_idx, axis=1
    )
    print(optical_flow_non_key_to_key_graph_points)
    # [1, N_non_key_graph ,3]
    optical_flow_non_key_to_key_graph_points = tf.squeeze(
        optical_flow_non_key_to_key_graph_points, axis=2
    )
    print(optical_flow_non_key_to_key_graph_points)
    # key frame graph point xyz
    # [1, N_key_graph ,3]
    xyz = tf.expand_dims(pc1_graph_point_xyz, 0)
    print('xyz: ', xyz)
    # non key frame graph point xyz
    # [1, N_non_key_graph, 3]
    xyz1 = tf.expand_dims(pc2_graph_point_xyz, 0)
    print('xyz1: ', xyz1)
    # [1, N_pc_non_key_graph, 300]
    t_features1 = feature_flow(
        xyz,
        xyz1,
        optical_flow_non_key_to_key_graph_points,
        tf.expand_dims(pc1_graph_feat, 0),
        n_sample=n_sample
    )
    # [N_pc_non_key_graph, 300]
    t_features1 = tf.squeeze(t_features1, axis=0)
    return t_features1


def feature_flow(xyz1, xyz2, flow_2_to_1, feature1, n_sample=4):
    '''
    :param xyz1: [batch_size, N1, 3]
    :param xyz2: [batch_size, N2, 3]
    :param flow_2_to_1: [batch_size, N2, 3]
    :param feature1: [batch_size, N1, c]
    :return: warped_feature: [batch_size, N2, c]
    '''
    # xyz1p = tf.Print(xyz1, [xyz1], "xyz1: ")
    # xyz2p = tf.Print(xyz2, [xyz2], "xyz2: ")
    # flow_2_to_1p = tf.Print(flow_2_to_1, [flow_2_to_1], "flow_2_to_1: ")
    xyz_2_in_1 = xyz2 + flow_2_to_1
    # xyz_2_in_1p = tf.Print(xyz_2_in_1, [xyz_2_in_1], "xyz_2_in_1: ")
    # idx=[batch_size, N2, n_sample]
    _, idx = knn_point(n_sample, xyz1, xyz_2_in_1)
    print('idx: ', idx)
    # shape = [batch_size, N2, n_sample, c]
    warped_feature = tf.gather(feature1, idx[0], axis=1)
    print(warped_feature)
    # shape = [batch_size, N2, c]
    warped_feature = tf.reduce_mean(warped_feature, axis=2)
    print(warped_feature)
    return warped_feature

def get_flownet3d_small_model(pc1_xyz, pc1_feat, pc2_xyz, pc2_feat, is_training, bn_decay=None):
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

    l0_xyz_f1 = pc1_xyz
    l0_points_f1 = pc1_feat
    l0_xyz_f2 = pc2_xyz
    l0_points_f2 = pc2_feat

    RADIUS1 = 0.5
    RADIUS2 = 1.0
    RADIUS3 = 2.0
    RADIUS4 = 4.0
    with tf.variable_scope('sa1') as scope:
        # Frame 1, Layer 1
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=1024, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        end_points['l1_indices_f1'] = l1_indices_f1

        # Frame 1, Layer 2
        l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        end_points['l2_indices_f1'] = l2_indices_f1

        scope.reuse_variables()
        # Frame 2, Layer 1
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=1024, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        # Frame 2, Layer 2
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    _, l2_points_f1_new = flow_embedding_module(l2_xyz_f1, l2_xyz_f2, l2_points_f1, l2_points_f2, radius=10.0, nsample=64, mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding', bn=True, pooling='max', knn=True, corr_func='concat')

    # Layer 3
    l3_xyz_f1, l3_points_f1, l3_indices_f1 = pointnet_sa_module(l2_xyz_f1, l2_points_f1_new, npoint=64, radius=RADIUS3, nsample=8, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    end_points['l3_indices_f1'] = l3_indices_f1

    # Layer 4
    l4_xyz_f1, l4_points_f1, l4_indices_f1 = pointnet_sa_module(l3_xyz_f1, l3_points_f1, npoint=16, radius=RADIUS4, nsample=8, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    end_points['l4_indices_f1'] = l4_indices_f1

    # Feature Propagation
    l3_feat_f1 = set_upconv_module(l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1, nsample=8, radius=2.4, mlp=[], mlp2=[256,256], scope='up_sa_layer1', is_training=is_training, bn_decay=bn_decay, knn=True)
    l2_feat_f1 = set_upconv_module(l2_xyz_f1, l3_xyz_f1, tf.concat(axis=-1, values=[l2_points_f1, l2_points_f1_new]), l3_feat_f1, nsample=8, radius=1.2, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer2', is_training=is_training, bn_decay=bn_decay, knn=True)
    l1_feat_f1 = set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_feat_f1, nsample=8, radius=0.6, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer3', is_training=is_training, bn_decay=bn_decay, knn=True)
    l0_feat_f1 = pointnet_fp_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_feat_f1, [256,256], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_feat_f1, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net

def get_flownet3d_large_model(pc1_xyz, pc1_feat, pc2_xyz, pc2_feat, is_training, bn_decay=None, reuse=False):
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
    #import tf_util
    print(tf_util)
    with tf.Graph().as_default():
        pc1_xyz = tf.zeros((32, 1024, 3))
        pc1_feat = tf.zeros((32, 1024, 4))
        pc2_xyz = tf.zeros((32, 1024, 3))
        pc2_feat = tf.zeros((32, 1024, 4))
        outputs = get_flownet3d_model(pc1_xyz, pc1_feat, pc2_xyz, pc2_feat, tf.constant(True))
        print(outputs)
