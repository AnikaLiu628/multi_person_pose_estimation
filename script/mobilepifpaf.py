import tensorflow as tf
import numpy as np

# from mobilenet_v1 import mobilenet_v1
# from mobilenet_v2 import mobilenet, training_scope
from shufflenet_v2 import ShuffleNetV2


class MobilePifPaf():
    def __init__(self, backbone, is_training, depth_multiplier=1.0, number_keypoints=17):
        self.backbone = backbone
        self.is_training = is_training
        self.depth_multiplier = depth_multiplier
        self.number_keypoints = number_keypoints

    def headnet(self, head_name, x,
                n_fields=None, n_vectors=None, n_scales=None,
                kernel_size=1, padding=0, dilation=1, is_training=False):

        with tf.variable_scope(head_name, reuse=tf.AUTO_REUSE):
            nets = []
            if is_training:
                x = tf.nn.dropout(x, keep_prob=0.5)
            out_features = n_fields * 4
            classes_x = tf.layers.conv2d(x, out_features, kernel_size, name='cls') # [b, h, w, 17*4]

            if not is_training:
                classes_x = tf.keras.activations.sigmoid(classes_x)

            classes_x = tf.nn.depth_to_space(classes_x, 2) # [b, h, w, 17]
            classes_x = tf.transpose(classes_x, [0, 3, 1, 2], name='class_out') # [b, 17, h, w]
            nets.append(classes_x)
            for n in range(n_vectors):
                reg_net = tf.layers.conv2d(x, 2 * out_features, kernel_size, name='reg' + str(n)) # [b, h, w, 17*4*2]
                regs_x = tf.nn.depth_to_space(reg_net, 2) # [b, h, w, 17*2]
                regs_x = tf.reshape(regs_x, [-1, regs_x.shape[1], regs_x.shape[2], regs_x.shape[3] // 2, 2]) # [b, h, w, 17, 2]
                regs_x = tf.transpose(regs_x, [0, 3, 4, 1, 2], name='regression_out') # [b, 17, 2, h, w]
                nets.append(regs_x)
            for n in range(n_vectors):
                spreads_net = tf.layers.conv2d(x, out_features, kernel_size, name='spr' + str(n)) # [b, h, w, 17*4]
                # spreads_net = tf.nn.leaky_relu(spreads_net + 2.0, alpha=0.01) - 2.0
                spreads_net = tf.nn.relu(spreads_net)
                regs_x_spread = tf.nn.depth_to_space(spreads_net, 2) # [b, h, w, 17]
                regs_x_spread = tf.transpose(regs_x_spread, [0, 3, 1, 2], name='spread_out') # [b, 17, h, w]
                nets.append(regs_x_spread)
            for n in range(n_scales):
                scale_net = tf.layers.conv2d(x, out_features, kernel_size, name='scl' + str(n))
                scale_net = tf.nn.relu(scale_net)
                scales_x = tf.nn.depth_to_space(scale_net, 2)
                scales_x = tf.transpose(scales_x, [0, 3, 1, 2], name='scale_out')
                nets.append(scales_x)
            

        return nets

    def build(self, features):
        pif_nfields = 17
        paf_nfields = 19
        pif_nvectors = 1
        paf_nvectors = 2
        pif_nscales = 1
        paf_nscales = 0

        # if self.backbone == 'mobilenet_v1':
        #     logits, end_points = mobilenet_v1(features,
        #                                       num_classes=False,
        #                                       is_training=self.is_training,
        #                                       depth_multiplier=self.depth_multiplier)
        #     backbone_end = end_points['Conv2d_13_pointwise']
        # elif self.backbone == 'mobilenet_v2':
        #     with tf.contrib.slim.arg_scope(training_scope(is_training=self.is_training)):
        #         logits, end_points = mobilenet(features,
        #                                        num_classes=False,
        #                                        depth_multiplier=self.depth_multiplier)

        #     backbone_end = end_points['layer_19']
        # elif self.backbone == 'shufflenet_v2':
        basenet = ShuffleNetV2(depth_multiplier=self.depth_multiplier,
                               is_training=self.is_training)
        end_points = basenet.build(features)
        backbone_end = end_points['base_net/out']

        pif = self.headnet('pif', backbone_end, pif_nfields, pif_nvectors, pif_nscales)
        paf = self.headnet('paf', backbone_end, paf_nfields, paf_nvectors, paf_nscales)

        end_points['PIF'] = pif
        end_points['PAF'] = paf
        end_points['outputs'] = [pif, paf]

        return end_points


def main(_):
    print('Rebuild graph...')
    model = MobilePifPaf(backbone='mobilenet_v1', is_training=False)

    inputs = tf.placeholder(tf.float32,
                            shape=(1, 360, 640, 3),
                            name='image')
    end_points = model.build(inputs)
    print(end_points['outputs'])
    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    PIF, PAF = sess.run(end_points['outputs'], feed_dict={inputs: np.zeros((1, 360, 640, 3))})
    print(PIF[0].shape, PIF[1].shape, PIF[2].shape, PIF[3].shape)
    print(PAF[0].shape, PAF[1].shape, PAF[2].shape, PAF[3].shape, PAF[4].shape)


if __name__ == '__main__':
    tf.app.run()