import tensorflow as tf
import numpy as np

from mobilenet_v1 import mobilenet_v1
from mobilenet_v2 import mobilenet, training_scope
from shufflenet_v2 import ShuffleNetV2
from network_mobilenet_thin import MobilenetNetworkThin


class MobilePifPaf():
    def __init__(self, backbone, is_training, depth_multiplier=0.5, number_keypoints=19):
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

    def build(self, features):
        n_heatmaps = 19
        paf_nfields = 19
        paf_nvectors = 2
        paf_nscales = 0
        # end_points=[]
        if self.backbone == 'mobilenet_v1':
            logits, end_points = mobilenet_v1(features, num_classes=False, is_training=self.is_training, depth_multiplier=self.depth_multiplier)
            backbone_end = end_points['Conv2d_13_pointwise']

        elif self.backbone == 'mobilenet_v2':
            with tf.contrib.slim.arg_scope(training_scope(is_training=self.is_training)):
                logits, end_points = mobilenet(features, num_classes=False, depth_multiplier=self.depth_multiplier)
            backbone_end = end_points['layer_19']

        elif self.backbone == 'shufflenet_v2':
            basenet = ShuffleNetV2(depth_multiplier=self.depth_multiplier, is_training=self.is_training)
            end_points = basenet.build(features)
            backbone_end = end_points['base_net/out']

        elif self.backbone == 'mobilenet_thin':
            out = MobilenetNetworkThin({'image': features}, conv_width=0.75, conv_width2=0.50, trainable=self.is_training)
            end_points = out.get_layer()
            hm = end_points['MConv_Stage6_L2_5']
            hm = tf.transpose(hm, [0, 3, 1, 2], name='hm_out')
            paf = end_points['MConv_Stage6_L1_5']
            paf = tf.transpose(paf, [0, 3, 1, 2], name='paf_out')

        if self.backbone == 'mobilenet_thin':
            # nets, hm, paf = self.headnet('paf', backbone_end, n_heatmaps, paf_nfields, paf_nvectors, paf_nscales)
            end_points['heat_map'] = hm
            end_points['PAF'] = paf
            # end_points['outputs'] = [nets]
        else:
            # nets, hm, paf = self.headnet('paf', backbone_end, n_heatmaps, paf_nfields, paf_nvectors, paf_nscales)
            ps1 = self.PixelShuffle(backbone_end, 2, 
                        scope='PixelShuffle1')

            paf_duc1 = self.DUC(ps1,
                        filters=512,
                        upscale_factor=2,
                        is_training=self.is_training,
                        scope='PAF_DUC1')

            paf_duc2 = self.DUC(paf_duc1,
                        filters=256,
                        upscale_factor=2,
                        is_training=self.is_training,
                        scope='PAF_DUC2')
            paf_duc2_pad = tf.pad(paf_duc2,
                            [[0, 0], [1, 1], [1, 1], [0, 0]],
                            name='duc2_padding')
            paf_conv_out = tf.layers.conv2d(paf_duc2_pad, 38,
                                        kernel_size=[3, 3],
                                        name='PAF_output')
            paf_conv_feature = tf.space_to_depth(paf_conv_out,
                                                block_size=int(4),
                                                data_format='NHWC',
                                                name='space_to_depth')
            concat_feat = tf.concat(values=[ps1, paf_conv_feature], axis=3, name='concat_feat')

            duc1 = self.DUC(concat_feat,
                        filters=512,
                        upscale_factor=2,
                        is_training=self.is_training,
                        scope='DUC1')
            duc2 = self.DUC(duc1,
                            filters=256,
                            upscale_factor=2,
                            is_training=self.is_training,
                            scope='DUC2')
            duc2_pad = tf.pad(duc2,
                            [[0, 0], [1, 1], [1, 1], [0, 0]],
                            name='duc2_padding')

            conv_out = tf.layers.conv2d(duc2_pad, self.number_keypoints,
                                        kernel_size=[3, 3],
                                        name='output')
            conv_out = tf.transpose(conv_out, [0, 3, 1, 2], name='hm_out')
            paf_conv_out = tf.transpose(paf_conv_out, [0, 3, 1, 2], name='paf_out')
            end_points['heat_map'] = conv_out
            end_points['PAF'] = paf_conv_out

        return end_points


def main(_):
    print('Rebuild graph...')
    model = MobilePifPaf(backbone='mobilenet_v1', is_training=False)

    inputs = tf.placeholder(tf.float32,
                            shape=(1, 384, 512, 3),
                            name='image')
    end_points = model.build(inputs)
    # print(end_points) #([class, paf1, paf2], class, paf)

    print(end_points['PAF'])
    # print(end_points['PAF'][0][1])

    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    end_points = sess.run(end_points, feed_dict={inputs: np.zeros((1, 384, 512, 3))})
    print(end_points['heat_map'].shape)
    # print(PAF[0][0])
    # print(PAF[1])


if __name__ == '__main__':
    tf.app.run()