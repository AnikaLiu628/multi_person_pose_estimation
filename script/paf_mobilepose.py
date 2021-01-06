import tensorflow as tf
import numpy as np
import time

from mobilenet import training_scope
from mobilenet_v1 import mobilenet_v1
from mobilenet_v2 import mobilenet as mobilenet_v2
from mobilenet_v3 import mobilenet as mobilenet_v3

from shufflenet_v2 import ShuffleNetV2
from network_mobilenet_thin import MobilenetNetworkThin

class PAFMobilePose():
    def __init__(self, backbone, is_training, depth_multiplier=1.0, number_keypoints=17, paf_block=1):
        self.backbone = backbone
        self.is_training = is_training
        self.depth_multiplier = depth_multiplier
        self.number_keypoints = number_keypoints
        self.paf_block = paf_block

    def PixelShuffle(self, I, r, scope='PixelShuffle'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            N, H, W, C = I.shape
            ps = tf.layers.conv2d_transpose(I, filters=int(C.value / r**2),
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding='same',
                                            use_bias=False,
                                            name='deconv')
            # ps = tf.depth_to_space(I, block_size=int(r),
            #                               data_format='NHWC',
            #                               name='depth_to_space')
        return ps

    def DUC(self, x, filters, upscale_factor, is_training, scope='DUC'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x_pad = tf.pad(x,
                           [[0, 0], [1, 1], [1, 1], [0, 0]],
                           name='x_padding')
            net = tf.layers.separable_conv2d(x_pad, filters,
                                             kernel_size=[3, 3],
                                             use_bias=False,
                                             name='sep_conv')
            net = tf.layers.batch_normalization(net,
                                                name='bn',
                                                training=is_training)
            net = tf.nn.relu(net)
            net = self.PixelShuffle(net, upscale_factor)
        return net

    def PAF(self, x, is_training, scope='PAF'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            paf_duc1 = self.DUC(x,
                                filters=512,
                                upscale_factor=2,
                                is_training=self.is_training,
                                scope='DUC1')
            paf_duc2 = self.DUC(paf_duc1,
                                filters=256,
                                upscale_factor=2,
                                is_training=self.is_training,
                                scope='DUC2')
            paf_duc2_pad = tf.pad(paf_duc2,
                                    [[0, 0], [1, 1], [1, 1], [0, 0]],
                                    name='duc2_padding')
            paf_conv_out = tf.layers.conv2d(paf_duc2_pad, 10,
                                            kernel_size=[3, 3],
                                            name='output')
            paf_conv_feat = tf.space_to_depth(paf_conv_out,
                                                block_size=int(4),
                                                data_format='NHWC',
                                                name='space_to_depth')
            concat_feat = tf.concat(values=[self.raw_feature_map, paf_conv_feat],
                                    axis=3,
                                    name='concat_feat')
        return paf_conv_out, concat_feat

    def FPN(self, x, is_training, scope='FPN'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            out1 = tf.layers.separable_conv2d(x, 128,
                                              kernel_size=[3, 3],
                                              strides=(2, 2),
                                              use_bias=False,
                                              name='sep_conv1')
            out = tf.layers.separable_conv2d(out1, 128,
                                             kernel_size=[3, 3],
                                             strides=(2, 2),
                                             use_bias=False,
                                             name='sep_conv2')
            out = tf.layers.separable_conv2d(out, 256,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='sep_conv3')
            out = self.DUC(out, 
                           filters=512, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC1')
            pad_out = tf.pad(out, 
                             [[0, 0], [1, 1], [1, 1], [0, 0]], 
                             name='DUC1_pad')
            con_out = tf.layers.separable_conv2d(out1, 512,
                                                 kernel_size=[1, 1],
                                                 use_bias=False,
                                                 name='sep_conv4')
            out = tf.concat(values=[con_out, pad_out], 
                            axis=3, 
                            name='concat_sp1_duc1pad')
            out = self.DUC(out, 
                           filters=256, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC2')
            pad_out = tf.pad(out, 
                             [[0, 0], [1, 1], [1, 1], [0, 0]], 
                             name='DUC2_pad')
            out = tf.concat(values=[pad_out, x], 
                            axis=3, 
                            name='concat_duc1pad_endpointout')
            out = tf.layers.separable_conv2d(out, 256,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='sep_conv5')
            out = self.PixelShuffle(out, 2, 
                                    scope='PixelShuffle1')
        return out

    def build(self, features):

        if self.backbone == 'mobilenet_thin_FPN':
            out = MobilenetNetworkThin({'image': features}, conv_width=0.75, conv_width2=0.50, trainable=self.is_training)
            end_points = out.get_layer()
            
            ###HEATMAP
            thin_hm = end_points['MConv_Stage6_L2_5']
            classes_hm1 = tf.layers.conv2d(thin_hm, 128, 3, strides=2, name='cls1')
            classes_hm2 = tf.layers.conv2d(classes_hm1, 256, 3, strides=2, name='cls2')
            con1_hm2 = tf.layers.conv2d(classes_hm2, 256, 1, name='1con2')
            duc_hm2 = self.DUC(con1_hm2, filters=512, upscale_factor=2, is_training=self.is_training, scope='DUC_hm2')
            pad_hm2 = tf.pad(duc_hm2, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad_hm2')
            con1_hm1 = tf.layers.conv2d(classes_hm1, 512, 1, name='1con1')
            concat_feat = tf.concat(values=[con1_hm1, pad_hm2], axis=3, name='concat_feat_p1')
            duc_hm1 = self.DUC(concat_feat, filters=256, upscale_factor=2, is_training=self.is_training, scope='DUC_hm1')
            pad_hm1 = tf.pad(duc_hm1, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad_hm1')
            hm_duc = tf.concat(values=[pad_hm1, thin_hm], axis=3, name='concat_feat_p2')
            hm_ch1 = tf.layers.conv2d(hm_duc, 128, kernel_size=[1, 1], name='hm_channel1')
            ps1 = self.PixelShuffle(hm_ch1, 2, scope='PixelShuffle1')
            hm_out = tf.layers.conv2d(ps1, 17, kernel_size=[1, 1], name='hm_conv') 
            hm = tf.transpose(hm_out, [0, 3, 1, 2], name='hm_out')

            ###PAF
            thin_paf = end_points['MConv_Stage6_L1_5']
            classes_paf1 = tf.layers.conv2d(thin_paf, 128, 3, strides=2, name='cls1_paf')
            classes_paf2 = tf.layers.conv2d(classes_paf1, 256, 3, strides=2, name='cls2_paf')
            con1_paf2 = tf.layers.conv2d(classes_paf2, 256, 1, name='1con2_paf')
            duc_paf2 = self.DUC(con1_paf2, filters=512, upscale_factor=2, is_training=self.is_training, scope='DUC_paf2')
            pad_paf2 = tf.pad(duc_paf2, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad_paf2')
            con1_paf1 = tf.layers.conv2d(classes_paf1, 512, 1, name='1con1_paf')
            concat_feat_paf = tf.concat(values=[con1_paf1, pad_paf2], axis=3, name='concat_feat_p1_paf')
            duc_paf1 = self.DUC(concat_feat_paf, filters=256, upscale_factor=2, is_training=self.is_training, scope='DUC_paf1')
            pad_paf1 = tf.pad(duc_paf1, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad_paf1')
            paf_duc = tf.concat(values=[pad_paf1, thin_paf], axis=3, name='concat_feat_p2_paf')
            paf_ch1 = tf.layers.conv2d(paf_duc, 256, kernel_size=[1, 1], name='paf_channel1')  
            ps2 = self.PixelShuffle(paf_ch1, 2, scope='PixelShuffle2')
            paf_out = tf.layers.conv2d(ps2, 36, kernel_size=[1, 1], name='paf_conv')  
            paf = tf.transpose(paf_out, [0, 3, 1, 2], name='paf_out')

            end_points['heat_map'] = hm
            end_points['PAF'] = paf
        
        elif self.backbone == 'mobilenetv1':
            logits, end_points = mobilenet_v1(features, 
                                              num_classes=False, 
                                              is_training=self.is_training, 
                                              depth_multiplier=self.depth_multiplier)

            backbone_end = end_points['Conv2d_13_pointwise']
            out = tf.layers.separable_conv2d(backbone_end, 64,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='hm_conv1')
            out = self.PixelShuffle(out, 
                                    4, 
                                    scope='PixelShuffle_hm')
            out = tf.layers.separable_conv2d(out, 32,
                                             kernel_size=[3, 3],
                                             use_bias=False,
                                             name='hm_conv2')
            out = self.DUC(out, 
                           filters=64, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC1_hm')
            out = tf.pad(out, 
                        [[0, 0], [1, 1], 
                         [1, 1], [0, 0]], 
                        name='pad_hm')
            out = self.DUC(out, 
                           filters=128, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC2_hm')
            hm_out = tf.layers.separable_conv2d(out, 17,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='hm_conv3')
            hm = tf.transpose(hm_out, [0, 3, 1, 2], name='hm_out')
            end_points['heat_map'] = hm

            out = tf.layers.separable_conv2d(backbone_end, 64,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='paf_conv1')
            out = self.PixelShuffle(out, 
                                    4, 
                                    scope='PixelShuffle_paf')
            out = tf.layers.separable_conv2d(out, 32,
                                             kernel_size=[3, 3],
                                             use_bias=False,
                                             name='paf_conv2')
            out = self.DUC(out, 
                           filters=64, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC1_paf')
            out = tf.pad(out, 
                        [[0, 0], [1, 1], 
                         [1, 1], [0, 0]], 
                        name='pad_paf')
            out = self.DUC(out, 
                           filters=128, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC2_paf')
            paf_out = tf.layers.separable_conv2d(out, 36,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='paf_conv3')
            paf = tf.transpose(paf_out, [0, 3, 1, 2], name='paf_out')
            end_points['PAF'] = paf

        elif self.backbone == 'mobilenetv2':
            with tf.contrib.slim.arg_scope(training_scope(is_training=self.is_training)):
                logits, end_points = mobilenet_v2(features, num_classes=False, depth_multiplier=self.depth_multiplier)
            backbone_end = end_points['layer_19']

            out = tf.layers.separable_conv2d(backbone_end, 64,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='hm_conv1')
            out = self.PixelShuffle(out, 
                                    4, 
                                    scope='PixelShuffle_hm')
            out = tf.layers.separable_conv2d(out, 32,
                                             kernel_size=[3, 3],
                                             use_bias=False,
                                             name='hm_conv2')
            out = self.DUC(out, 
                           filters=64, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC1_hm')
            out = tf.pad(out, 
                        [[0, 0], [1, 1], 
                         [1, 1], [0, 0]], 
                        name='pad_hm')
            out = self.DUC(out, 
                           filters=128, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC2_hm')
            hm_out = tf.layers.separable_conv2d(out, 17,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='hm_conv3')
            hm = tf.transpose(hm_out, [0, 3, 1, 2], name='hm_out')
            end_points['heat_map'] = hm

            out = tf.layers.separable_conv2d(backbone_end, 64,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='paf_conv1')
            out = self.PixelShuffle(out, 
                                    4, 
                                    scope='PixelShuffle_paf')
            out = tf.layers.separable_conv2d(out, 32,
                                             kernel_size=[3, 3],
                                             use_bias=False,
                                             name='paf_conv2')
            out = self.DUC(out, 
                           filters=64, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC1_paf')
            out = tf.pad(out, 
                        [[0, 0], [1, 1], 
                         [1, 1], [0, 0]], 
                        name='pad_paf')
            out = self.DUC(out, 
                           filters=128, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC2_paf')
            paf_out = tf.layers.separable_conv2d(out, 36,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='paf_conv3')
            paf = tf.transpose(paf_out, [0, 3, 1, 2], name='paf_out')
            end_points['PAF'] = paf

        elif self.backbone == 'mobilenetv3':
            with tf.contrib.slim.arg_scope(training_scope(is_training=self.is_training)):
                logits, end_points = mobilenet_v3(features,
                                               num_classes=False,
                                               depth_multiplier=self.depth_multiplier)


            backbone_end = end_points['layer_17']
            out = tf.layers.separable_conv2d(backbone_end, 64,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='hm_conv1')
            out = self.PixelShuffle(out, 
                                    4, 
                                    scope='PixelShuffle_hm')
            out = tf.layers.separable_conv2d(out, 32,
                                             kernel_size=[3, 3],
                                             use_bias=False,
                                             name='hm_conv2')
            out = self.DUC(out, 
                           filters=64, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC1_hm')
            out = tf.pad(out, 
                        [[0, 0], [1, 1], 
                         [1, 1], [0, 0]], 
                        name='pad_hm')
            out = self.DUC(out, 
                           filters=128, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC2_hm')
            hm_out = tf.layers.separable_conv2d(out, 17,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='hm_conv3')
            hm = tf.transpose(hm_out, [0, 3, 1, 2], name='hm_out')
            end_points['heat_map'] = hm

            out = tf.layers.separable_conv2d(backbone_end, 64,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='paf_conv1')
            out = self.PixelShuffle(out, 
                                    4, 
                                    scope='PixelShuffle_paf')
            out = tf.layers.separable_conv2d(out, 32,
                                             kernel_size=[3, 3],
                                             use_bias=False,
                                             name='paf_conv2')
            out = self.DUC(out, 
                           filters=64, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC1_paf')
            out = tf.pad(out, 
                        [[0, 0], [1, 1], 
                         [1, 1], [0, 0]], 
                        name='pad_paf')
            out = self.DUC(out, 
                           filters=128, 
                           upscale_factor=2, 
                           is_training=self.is_training, 
                           scope='DUC2_paf')
            paf_out = tf.layers.separable_conv2d(out, 36,
                                             kernel_size=[1, 1],
                                             use_bias=False,
                                             name='paf_conv3')
            paf = tf.transpose(paf_out, [0, 3, 1, 2], name='paf_out')
            end_points['PAF'] = paf

        elif self.backbone == 'shufflenet_v2':
            basenet = ShuffleNetV2(depth_multiplier=self.depth_multiplier, is_training=self.is_training)
            end_points = basenet.build(features)
            backbone_end = end_points['base_net/out']
            out = self.PixelShuffle(backbone_end, 4, scope='PixelShuffle_paf')
            out = tf.layers.conv2d(out, 36, kernel_size=[1, 1], name='cls_paf')
            paf = tf.transpose(out, [0, 3, 1, 2], name='paf_out')

            out = self.PixelShuffle(backbone_end, 4, scope='PixelShuffle_hm')
            out = tf.layers.conv2d(out, 17, kernel_size=[1, 1], name='cls_hm')
            hm = tf.transpose(out, [0, 3, 1, 2], name='hm_out')
            
            end_points['heat_map'] = hm
            end_points['PAF'] = paf

        return end_points

def main(_):
    print('Rebuild graph...')
    
    model = PAFMobilePose(backbone='small_mobilenet_thin_FPN', is_training=True)

    inputs = tf.compat.v1.placeholder(tf.float32,
                            shape=(1, 368, 432, 3),
                            name='image')

    end_points = model.build(inputs)

    print(end_points['PAF']) #36
    st_time = time.time()
    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    end_points = sess.run(end_points, feed_dict={inputs: np.zeros((1, 368, 432, 3))})
    print(end_points['heat_map'].shape) #17
    print(end_points['PAF'].shape)
    print(time.time()-st_time)

if __name__ == '__main__':
    tf.compat.v1.app.run()