import tensorflow as tf
import numpy as np
import time

from mobilenet_v1 import mobilenet_v1
from mobilenet_v2 import mobilenet, training_scope
from shufflenet_v2 import ShuffleNetV2
from network_mobilenet_thin import MobilenetNetworkThin
from hrnet import HRNet
from net.model import HRNet as preHRnet
# from pre_hrnet.model import HRNet 


class MobilePaf():
    def __init__(self, backbone, is_training, depth_multiplier=0.5, number_keypoints=17):
        self.backbone = backbone
        self.is_training = is_training
        self.depth_multiplier = depth_multiplier
        self.number_keypoints = number_keypoints

    def headnet(self, head_name, x, n_hm=None,
                n_fields=None, n_vectors=None, n_scales=None,
                kernel_size=1, padding=0, dilation=1, is_training=False):

        with tf.variable_scope(head_name, reuse=tf.AUTO_REUSE):
            nets = []
            if is_training:
                x = tf.nn.dropout(x, keep_prob=0.5)
            hm_out_features = n_hm * 16
            paf_out_features = n_fields * 16
            classes_x = tf.layers.conv2d(x, hm_out_features, kernel_size, name='cls1')
            classes_x = tf.layers.conv2d(classes_x, hm_out_features, kernel_size, name='cls2')
            if not is_training:
                classes_x = tf.keras.activations.sigmoid(classes_x)

            classes_x = tf.nn.depth_to_space(classes_x, 2)
            classes_x = tf.nn.depth_to_space(classes_x, 2)
            classes_x = tf.transpose(classes_x, [0, 3, 1, 2], name='class_out')
            nets.append(classes_x)

            for n in range(n_vectors):
                reg_net = tf.layers.conv2d(x, 2 * paf_out_features, kernel_size, name='reg1_' + str(n))
                reg_net = tf.layers.conv2d(reg_net, 2 * paf_out_features, kernel_size, name='reg2_' + str(n))
                regs_x = tf.nn.depth_to_space(reg_net, 2)
                regs_x = tf.nn.depth_to_space(regs_x, 2)
                regs_x = tf.reshape(regs_x, [-1, regs_x.shape[1], regs_x.shape[2], regs_x.shape[3]])
                regs_x = tf.transpose(regs_x, [0, 3, 1, 2], name='regression_out')
                nets.append(regs_x)
                
        return nets, classes_x, regs_x

    def PixelShuffle(self, I, r, scope='PixelShuffle'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            N, H, W, C = I.shape
            # ps = tf.layers.conv2d_transpose(I, filters=int(C.value / r**2),
            #                                 kernel_size=(3, 3),
            #                                 strides=(2, 2),
            #                                 padding='same',
            #                                 use_bias=False,
            #                                 name='deconv')
            ps = tf.depth_to_space(I, block_size=int(r),
                                          data_format='NHWC',
                                          name='depth_to_space')
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

    def HR_BasicBlock(self, x, filters, is_training, scope='HR_BasicBlock'):
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
            
        return net


    def build(self, features):
        n_heatmaps = 17
        paf_nfields = 18
        paf_nvectors = 2
        paf_nscales = 0
        if self.backbone == 'mobilenet_v1':
            logits, end_points = mobilenet_v1(features, num_classes=False, is_training=self.is_training, depth_multiplier=self.depth_multiplier)
            backbone_end = end_points['Conv2d_13_pointwise'] #1, 36, 46, 54
            nets = self.headnet('paf', backbone_end, n_heatmaps, paf_nfields, paf_nvectors, paf_nscales)

            end_points['PAF'] = nets
            end_points['outputs'] = [nets]
        elif self.backbone == 'mobilenet_v2':
            with tf.contrib.slim.arg_scope(training_scope(is_training=self.is_training)):
                logits, end_points = mobilenet(features, num_classes=False, depth_multiplier=self.depth_multiplier)
            backbone_end = end_points['layer_19']
            nets = self.headnet('paf', backbone_end, n_heatmaps, paf_nfields, paf_nvectors, paf_nscales)

            end_points['PAF'] = nets
            end_points['outputs'] = [nets]
        elif self.backbone == 'shufflenet_v2':
            basenet = ShuffleNetV2(depth_multiplier=self.depth_multiplier, is_training=self.is_training)
            end_points = basenet.build(features)
            backbone_end = end_points['base_net/out']
            nets = self.headnet('paf', backbone_end, n_heatmaps, paf_nfields, paf_nvectors, paf_nscales)

            end_points['PAF'] = nets
            end_points['outputs'] = [nets]

        elif self.backbone == 'mobilenet_thin_FPN':
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
            hm_out = tf.layers.conv2d(hm_duc, 17, kernel_size=[1, 1], name='hm_conv') 

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
            paf_out = tf.layers.conv2d(paf_duc, 36, kernel_size=[1, 1], name='paf_conv')  

            end_points['heat_map'] = hm
            end_points['PAF'] = paf

        elif self.backbone == 'mobilenet_thin_add_more_layers':
            out = MobilenetNetworkThin({'image': features}, conv_width=0.75, conv_width2=0.50, trainable=self.is_training)
            end_points = out.get_layer()
            
            ###HEATMAP
            thin_hm = end_points['MConv_Stage6_L2_5']
            s2d_hm = tf.space_to_depth(thin_hm, block_size=int(2), data_format='NHWC', name='space_to_depth_hm')
            hm_duc = self.DUC(s2d_hm, filters=512, upscale_factor=2, is_training=self.is_training, scope='DUC_hm')
            hm_out = tf.layers.conv2d(hm_duc, 17, kernel_size=[1, 1], name='hm_conv')  
            hm = tf.transpose(hm_out, [0, 3, 1, 2], name='hm_out')
            ###PAF
            thin_paf = end_points['MConv_Stage6_L1_5']
            s2d_paf = tf.space_to_depth(thin_paf, block_size=int(2), data_format='NHWC', name='space_to_depth_paf')
            paf_duc = self.DUC(s2d_paf, filters=512, upscale_factor=2, is_training=self.is_training, scope='DUC_paf')
            paf_out = tf.layers.conv2d(paf_duc, 36, kernel_size=[1, 1], name='paf_conv')  
            paf = tf.transpose(paf_out, [0, 3, 1, 2], name='paf_out')

            end_points['heat_map'] = hm
            end_points['PAF'] = paf
             
        elif self.backbone == 'mobilenet_thin_out4':
            out = MobilenetNetworkThin({'image': features}, conv_width=0.75, conv_width2=0.50, trainable=self.is_training)
            end_points = out.get_layer()

            thin_hm = end_points['MConv_Stage6_L2_5']
            hm_ch1 = tf.layers.conv2d(thin_hm, 128, kernel_size=[1, 1], name='hm_channel1')  
            ps1 = self.PixelShuffle(hm_ch1, 2, scope='PixelShuffle1')
            hm_out = tf.layers.conv2d(ps1, 17, kernel_size=[1, 1], name='hm_channel2')  
            hm = tf.transpose(hm_out, [0, 3, 1, 2], name='hm_out')
            
            thin_paf = end_points['MConv_Stage6_L1_5']
            paf_ch1 = tf.layers.conv2d(thin_paf, 256, kernel_size=[1, 1], name='paf_channel1')  
            ps2 = self.PixelShuffle(paf_ch1, 2, scope='PixelShuffle2')
            paf_out = tf.layers.conv2d(ps2, 36, kernel_size=[1, 1], name='paf_channel2')  
            paf = tf.transpose(paf_out, [0, 3, 1, 2], name='paf_out')

            end_points['heat_map'] = hm
            end_points['PAF'] = paf

        elif self.backbone == 'hrnet':
            end_points = dict()
            out = HRNet(features)
            backbone_end = out
            s2d_1 = tf.space_to_depth(backbone_end,
                                        block_size=int(4),
                                        data_format='NHWC',
                                        name='space_to_depth_1')
            paf_cov1 = tf.layers.conv2d(s2d_1, 64, #38
                                        kernel_size=[1, 1],
                                        name='paf_cov1')
            s2d_2 = tf.space_to_depth(paf_cov1,
                                        block_size=int(2),
                                        data_format='NHWC',
                                        name='space_to_depth_2')
            paf = tf.layers.conv2d(s2d_2, 36, #38
                                    kernel_size=[1, 1],
                                    name='paf_conv')
            concat_feat = tf.concat(values=[s2d_1, paf_cov1], axis=3, name='concat_feat')

            ps1 = self.PixelShuffle(concat_feat, 2, 
                        scope='PixelShuffle1')
            hm_duc1 = self.DUC(ps1,
                                filters=512,
                                upscale_factor=2,
                                is_training=self.is_training,
                                scope='DUC1')
            hm_duc2 = self.DUC(hm_duc1,
                                filters=256,
                                upscale_factor=2,
                                is_training=self.is_training,
                                scope='DUC2')
            s2d_3 = tf.space_to_depth(paf_cov1,
                                        block_size=int(2),
                                        data_format='NHWC',
                                        name='space_to_depth_3')
            hm = tf.layers.conv2d(s2d_2, 17, #38
                                        kernel_size=[1, 1],
                                        name='hm_conv')
            hm_out = tf.transpose(hm, [0, 3, 1, 2], name='hm_out')
            paf_out = tf.transpose(paf, [0, 3, 1, 2], name='paf_out')
            end_points['heat_map'] = hm_out
            end_points['PAF'] = paf_out

        elif self.backbone == 'hrnet_tiny':
            end_points = dict()
            out = HRNet(features)
            backbone_end = out
            conv_paf1 = tf.layers.conv2d(backbone_end, 128, 3, strides=2, name='paf_conv1')
            conv_paf2 = tf.layers.conv2d(conv_paf1, 128, 3, strides=2, name='paf_conv2')
            conv_paf3 = tf.layers.conv2d(conv_paf2, 128, 3, strides=1, name='paf_conv3')
            conv_paf4 = tf.layers.conv2d(conv_paf3, 128, 3, strides=2, name='paf_conv4')
            pad_paf = tf.pad(conv_paf4, [[0, 0], [1, 1], [1, 1], [0, 0]], name='paf_pad')
            paf = tf.layers.conv2d(pad_paf, 36, kernel_size=[1, 1], name='paf_conv')

            conv_hm1 = tf.layers.conv2d(backbone_end, 128, 3, strides=2, name='hm_conv1')
            conv_hm2 = tf.layers.conv2d(conv_hm1, 128, 3, strides=2, name='hm_conv2')
            conv_hm3 = tf.layers.conv2d(conv_hm2, 128, 3, strides=1, name='hm_conv3')
            conv_hm4 = tf.layers.conv2d(conv_hm3, 128, 3, strides=2, name='hm_conv4')
            pad_hm = tf.pad(conv_hm4, [[0, 0], [1, 1], [1, 1], [0, 0]], name='hm_pad')
            hm = tf.layers.conv2d(pad_hm, 17, kernel_size=[1, 1], name='hm_conv')

            hm_out = tf.transpose(hm, [0, 3, 1, 2], name='hm_out')
            paf_out = tf.transpose(paf, [0, 3, 1, 2], name='paf_out')
            end_points['heat_map'] = hm_out
            end_points['PAF'] = paf_out

        elif self.backbone == 'higher_hrnet':
            is_training = True
            end_points = dict()
            backbone_end = HRNet(features)
            #Downsampling
            downsample1 = tf.layers.conv2d(backbone_end, 64, 1, strides=2, name='downsample_1')
            bn_downsample1 = tf.layers.batch_normalization(downsample1, name='downsample_1_bn', training=is_training)
            downsample1 = tf.nn.relu(bn_downsample1)
            downsample2 = tf.layers.conv2d(downsample1, 64, 1, strides=2, name='downsample_2')
            bn_downsample2 = tf.layers.batch_normalization(downsample2, name='downsample2_bn', training=is_training)
            downsample2 = tf.keras.activations.relu(bn_downsample2) #1/4 input size (1, 92, 108, 128)
            conv_paf3 = tf.layers.conv2d(downsample2, 64, 1, strides=2, name='paf_conv3')
            bn_downsample3 = tf.layers.batch_normalization(conv_paf3, name='downsample3_bn', training=is_training)
            downsample3 = tf.keras.activations.relu(bn_downsample3) #(1, 46, 54, 128)
            
            #paf layer
            paf_final_conv1 = tf.layers.conv2d(downsample3, 192, 1, strides=1, name='final_conv1_paf')
            paf_final_conv2 = tf.layers.conv2d(paf_final_conv1, 192, 1, strides=1, name='final_conv2_paf')
            paf_output = tf.concat(values=[paf_final_conv2, downsample3], axis=3, name='ouput_paf')
            paf_adjust = tf.layers.conv2d(paf_output, 36, 1, strides=1, name='adjust_paf')
            
            #FinalLayer
            final_conv1 = tf.layers.conv2d(downsample2, 192, 1, strides=1, name='final_conv1')
            final_conv2 = tf.layers.conv2d(final_conv1, 192, 1, strides=1, name='final_conv2')
            conc_final_conv2 = tf.concat(values=[final_conv2, downsample2], axis=3, name='concat_finalconv2_downsam2')
            #Deconv block
            ps1 = self.DUC(conc_final_conv2, filters=32, upscale_factor=2, is_training=self.is_training, scope='DUC1')
            ps2 = self.DUC(ps1, filters=32, upscale_factor=2, is_training=self.is_training, scope='DUC2')
            s2d_1 = tf.space_to_depth(ps2, block_size=int(4), data_format='NHWC', name='space_to_depth_1')
            s2d_2 = tf.space_to_depth(s2d_1, block_size=int(2), data_format='NHWC', name='space_to_depth_2')
            #BasicLayer
            basic1 = self.HR_BasicBlock(s2d_2, filters=32, is_training=self.is_training, scope='basic_block1')
            basic2 = self.HR_BasicBlock(basic1, filters=32, is_training=self.is_training, scope='basic_block2')
            basic3 = self.HR_BasicBlock(basic2, filters=32, is_training=self.is_training, scope='basic_block3')
            basic4 = self.HR_BasicBlock(basic3, filters=32, is_training=self.is_training, scope='basic_block4')
            basic4 = tf.nn.relu(basic4)
            pad_basic4 = tf.pad(basic4, [[0, 0], [1, 1], [1, 1], [0, 0]], name='basic4_padding')
            adjust = tf.layers.conv2d(pad_basic4, 17, 3, strides=1, name='adjust')
            
            hm_out = tf.transpose(adjust, [0, 3, 1, 2], name='hm_out')
            paf_out = tf.transpose(paf_adjust, [0, 3, 1, 2], name='paf_out')
            end_points['heat_map'] = hm_out
            end_points['PAF'] = paf_out

        elif self.backbone == 'pre_hrnet':
            end_points = dict()
            hrnet = preHRnet(cfgfile='/cfgs/w30_s4.cfg')
            backbone_end = hrnet.forward_train(features)
            print(backbone_end)

        return end_points


def main(_):
    print('Rebuild graph...')
    
    model = MobilePaf(backbone='higher_hrnet', is_training=True)

    inputs = tf.placeholder(tf.float32,
                            shape=(1, 368, 432, 3),
                            name='image')

    end_points = model.build(inputs)

    print(end_points['PAF'])
    st_time = 0
    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    end_points = sess.run(end_points, feed_dict={inputs: np.zeros((1, 368, 432, 3))})
    print(end_points['heat_map'].shape)
    print(end_points['PAF'].shape)
    print(time.time()-st_time)

if __name__ == '__main__':
    tf.app.run()