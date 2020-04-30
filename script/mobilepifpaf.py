import tensorflow as tf
import numpy as np

from mobilenet_v1 import mobilenet_v1
from mobilenet_v2 import mobilenet, training_scope
from shufflenet_v2 import ShuffleNetV2
from network_mobilenet_thin import MobilenetNetworkThin


class MobilePifPaf():
    def __init__(self, backbone, is_training, depth_multiplier=1.0, number_keypoints=18):
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

    def build(self, features):
        n_heatmaps = 18
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
            # end_points = basenet.build(features)
            end_points = out.get_layer()
            # end_points['Openpose/concat_stage7'] = backbone_end
            # print(end_points[-1])
            # backbone_end = end_points['Openpose/concat_stage7']
            # print(end_points)
            hm = end_points['MConv_Stage6_L2_5']
            hm = tf.transpose(hm, [0, 3, 1, 2], name='hm_out')
            paf = end_points['MConv_Stage6_L1_5']
            paf = tf.transpose(paf, [0, 3, 1, 2], name='paf_out')
            # print(hm, paf)

        # nets, hm, paf = self.headnet('paf', backbone_end, n_heatmaps, paf_nfields, paf_nvectors, paf_nscales)
        end_points['heat_map'] = hm
        end_points['PAF'] = paf
        # end_points['outputs'] = [nets]

        return end_points


def main(_):
    print('Rebuild graph...')
    model = MobilePifPaf(backbone='mobilenet_thin', is_training=False)

    inputs = tf.placeholder(tf.float32,
                            shape=(1, 368, 432, 3),
                            name='image')
    end_points = model.build(inputs)
    # print(end_points) #([class, paf1, paf2], class, paf)

    print(end_points['PAF'])
    # print(end_points['PAF'][0][1])

    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    end_points = sess.run(end_points, feed_dict={inputs: np.zeros((1, 368, 432, 3))})
    print(end_points['heat_map'].shape)
    # print(PAF[0][0])
    # print(PAF[1])


if __name__ == '__main__':
    tf.app.run()