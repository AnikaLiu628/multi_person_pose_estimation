import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def channel_shuffle(x, groups):
    # print('x;===========',x.shape)
    N, H, W, C = x.shape
    channels_per_group = C // groups

    out_net = tf.reshape(x, [-1, H, W, groups, channels_per_group])
    out_net = tf.transpose(out_net, [0, 1, 2, 4, 3])
    out_net = tf.reshape(out_net, [-1, H, W, C])

    return out_net


class InvertedResidual():
    def __init__(self, outputs, stride, is_training=False, scope='invres'):
        self.stride = stride
        self.branch_features = outputs // 2
        self.is_training = is_training
        self.scope = scope

    def get_filters(self, filters, kernel_size, w_init=tf.initializers.glorot_normal, name=None, suffix=''):
        filters = tf.get_variable(name='%s%s/depthwise_kernel' %(name, suffix),
                                              shape=[kernel_size[0],kernel_size[1],filters,1],
                                              initializer=w_init,
                                              dtype=tf.float32)
        return filters

    def build(self, x):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if self.stride > 1:
                with tf.variable_scope('branch1', reuse=tf.AUTO_REUSE):
                    net = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
                    branch1 = tf.nn.depthwise_conv2d(net,
                                                     self.get_filters(net.shape[-1], [3, 3],
                                                        name='depthwise1'),
                                                     strides=[1, self.stride, self.stride, 1],
                                                     padding='VALID',
                                                     name='depthwise1')
                    branch1 = tf.layers.batch_normalization(branch1,
                                                            name='bn1',
                                                            training=self.is_training)
                    branch1 = tf.layers.conv2d(branch1, self.branch_features, [1, 1],
                                               strides=1, padding='valid',
                                               use_bias=False, name='conv1')
                    branch1 = tf.layers.batch_normalization(branch1,
                                                            name='bn2',
                                                            training=self.is_training)
                    branch1 = tf.nn.relu(branch1)
                x2 = x
            else:
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=3)
                branch1 = x1

            with tf.variable_scope('branch2', reuse=tf.AUTO_REUSE):
                branch2 = tf.layers.conv2d(x2, self.branch_features, [1, 1],
                                           strides=1, padding='valid',
                                           use_bias=False, name='conv1')
                branch2 = tf.layers.batch_normalization(branch2,
                                                        name='bn1',
                                                        training=self.is_training)
                branch2 = tf.nn.relu(branch2)
                branch2 = tf.pad(branch2, [[0, 0], [1, 1], [1, 1], [0, 0]])
                branch2 = tf.nn.depthwise_conv2d(branch2,
                                                 self.get_filters(branch2.shape[-1], [3, 3],
                                                    name='depthwise1'),
                                                 strides=[1, self.stride, self.stride, 1],
                                                 padding='VALID',
                                                 name='depthwise1')
                branch2 = tf.layers.batch_normalization(branch2,
                                                        name='bn2',
                                                        training=self.is_training)
                branch2 = tf.layers.conv2d(branch2, self.branch_features, [1, 1],
                                           strides=1, padding='valid',
                                           use_bias=False, name='conv2')
                branch2 = tf.layers.batch_normalization(branch2,
                                                        name='bn3',
                                                        training=self.is_training)
                branch2 = tf.nn.relu(branch2)

            out_net = tf.concat([branch1, branch2], axis=3)
            out_net = channel_shuffle(out_net, 2)

        return out_net


class ShuffleNetV2():
    def __init__(self, depth_multiplier=1.0, is_training=False):
        if depth_multiplier == 0.5:
            self.stage_out_channels = [24, 48, 96, 192, 1024]
        elif depth_multiplier == 1.0:
            self.stage_out_channels = [24, 116, 232, 464, 1024]
        elif depth_multiplier == 1.5:
            self.stage_out_channels = [24, 176, 352, 704, 1024]
        elif depth_multiplier == 2.0:
            self.stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise ValueError(
                '--depth_multiplier invalid.'
                'Please select [0.5, 1.0, 1.5, 2.0]'
            )
        self.stages_repeats = [4, 8, 4]
        self.is_training = is_training


    def build(self, x):
        end_points = {}
        with tf.variable_scope('shufflenet_v2', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
                net = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
                net = tf.layers.conv2d(net, self.stage_out_channels[0], [3, 3],
                                       strides=2,
                                       padding='valid',
                                       use_bias=False,
                                       name='conv1')
                net = tf.layers.batch_normalization(net,
                                                    name='bn1',
                                                    training=self.is_training)
                net = tf.nn.relu(net)
            stage_names = ['2', '3', '4']
            for tag, repeats, output_channels in zip(stage_names,
                    self.stages_repeats, self.stage_out_channels[1:]):

                with tf.variable_scope('stage' + tag, reuse=tf.AUTO_REUSE):
                    net = InvertedResidual(output_channels, 2, self.is_training, 'invres0').build(net)
                    for i in range(1, repeats):
                        net = InvertedResidual(output_channels, 1, self.is_training, 'invres' + str(i)).build(net)

            with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
                net = tf.layers.conv2d(net, self.stage_out_channels[-1], [1, 1],
                                       strides=1,
                                       padding='valid',
                                       use_bias=False,
                                       name='conv5')
                net = tf.layers.batch_normalization(net,
                                                    name='bn5',
                                                    training=self.is_training)
                net = tf.nn.relu(net)
                end_points['base_net/out'] = net
        return end_points


def main(_):
    print('Rebuild graph...')
    model = ShuffleNetV2(depth_multiplier=1.0, is_training=False)

    inputs = tf.placeholder(tf.float32,
                            shape=(1, 256, 320, 3),
                            name='image')
    end_point = model.build(inputs)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    saver.save(sess, 'ShuffleNetV2/shufflenet_v2_1.0_360_640')
    output = sess.run(end_point, feed_dict={inputs: np.zeros((1, 256, 320, 3))})
    print(output)


if __name__ == '__main__':
    tf.app.run()