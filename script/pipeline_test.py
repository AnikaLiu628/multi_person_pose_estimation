import logging
from functools import partial
import math

import cv2
import numpy as np
import tensorflow as tf

from preprocesses import Preprocess


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string(
    'dataset_path',
    '../data/coco_mp_keypoints_train_100.record-00000-of-00001',
    'Training data'
)
flags.DEFINE_string(
    'validationset_path',
    '/datasets/coco/intermediate/coco_keypoints_val.record-00000-of-00001',
    'Validation data'
)
flags.DEFINE_string(
    'output_model_path',
    '/workspace/pose_estimation/models/PE_MOBILENET_V1_0.5_320_256_1',
    'Path of output human pose model'
)
flags.DEFINE_string(
    'pretrained_model_path',
    '/workspace/pose_estimation/models/pretrained_models/mobilenet_v1_0.5_128_2018_08_02/mobilenet_v1_0.5_128.ckpt',
    'Path of pretrained model(ckpt)'
)
flags.DEFINE_string(
    'loss_fn',
    'MSE',
    'Loss function in [MSE, softmax, center, focal, inv_focal, arcface]'
)
flags.DEFINE_float(
    'layer_depth_multiplier',
    0.5,
    'Depth multiplier of mobilenetv1 architecture'
)
flags.DEFINE_integer(
    'batch_size',
    1,
    'Size of batch data'
)
flags.DEFINE_boolean(
    'pretrained_model',
    False,
    'Use pretrained model or not'
)
flags.DEFINE_boolean(
    'data_augmentation',
    True,
    'Add data augmentation to preprocess'
)
flags.DEFINE_string(
    'optimizer',
    'Adam',
    'Optimizer in [Momentum, Adagrad, Adam, RMSProp, Nadam]'
)
flags.DEFINE_float(
    'learning_rate',
    0.001,
    'Learning rate for training process'
)
flags.DEFINE_integer(
    'decay_steps',
    1000000000,
    'Decay steps of learning rate'
)
flags.DEFINE_float(
    'decay_factor',
    0.1,
    'Decay factor of learning rate'
)
flags.DEFINE_integer(
    'training_steps',
    1000000,
    'Train n steps'
)
flags.DEFINE_integer(
    'validation_interval',
    1,
    'Evaluate validation loss for every n steps'
)
flags.DEFINE_integer(
    'validation_batch_size',
    1,
    'Size of batch data'
)

FLAGS = flags.FLAGS
pretrained_model_steps = 13109116


def _parse_function(example_proto):
    features = {'image/height': tf.FixedLenFeature((), tf.int64),
                'image/width': tf.FixedLenFeature((), tf.int64),
                'image/filename': tf.FixedLenFeature((), tf.string),
                'image/encoded': tf.FixedLenFeature((), tf.string),
                'image/format': tf.FixedLenFeature((), tf.string),
                'image/human/bbox/xmin': tf.VarLenFeature(tf.int64),
                'image/human/bbox/xmax': tf.VarLenFeature(tf.int64),
                'image/human/bbox/ymin': tf.VarLenFeature(tf.int64),
                'image/human/bbox/ymax': tf.VarLenFeature(tf.int64),
                'image/human/num_keypoints': tf.VarLenFeature(tf.int64),
                'image/human/keypoints/x': tf.VarLenFeature(tf.int64),
                'image/human/keypoints/y': tf.VarLenFeature(tf.int64),
                'image/human/keypoints/v': tf.VarLenFeature(tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['image/decoded'] = tf.image.decode_image(
        parsed_features['image/encoded'],
        channels=3
    )
    parsed_features['image/human'] = parsed_features['image/decoded']
    bx1 = tf.sparse_tensor_to_dense(parsed_features['image/human/bbox/xmin'], default_value=0)
    parsed_features['image/human/bbox/xmin'] = bx1
    bx2 = tf.sparse_tensor_to_dense(parsed_features['image/human/bbox/xmax'], default_value=0)
    parsed_features['image/human/bbox/xmax'] = bx2
    by1 = tf.sparse_tensor_to_dense(parsed_features['image/human/bbox/ymin'], default_value=0)
    parsed_features['image/human/bbox/ymin'] = by1
    by2 = tf.sparse_tensor_to_dense(parsed_features['image/human/bbox/ymax'], default_value=0)
    parsed_features['image/human/bbox/ymax'] = by2

    xs = tf.sparse_tensor_to_dense(parsed_features['image/human/keypoints/x'], default_value=0)
    parsed_features['image/human/keypoints/x'] = xs
    ys = tf.sparse_tensor_to_dense(parsed_features['image/human/keypoints/y'], default_value=0)
    parsed_features['image/human/keypoints/y'] = ys
    vs = tf.sparse_tensor_to_dense(parsed_features['image/human/keypoints/v'], default_value=0)
    parsed_features['image/human/keypoints/v'] = vs
    nkp = tf.sparse_tensor_to_dense(parsed_features['image/human/num_keypoints'], default_value=0)
    parsed_features['image/human/num_keypoints'] = nkp
    return parsed_features


def _preprocess_function(parsed_features, params={}):
    
    w = 640
    h = 360
    if not params['do_data_augmentation']:
        parsed_features['image/human'] = parsed_features['image/decoded']
    else:
        parsed_features['image/human'] = parsed_features['image/human']
    parsed_features['image/human'].set_shape([None, None, 3])

    parsed_features['image/human/resized'] = tf.image.resize_images(parsed_features['image/human'], [h, w], method=tf.image.ResizeMethod.AREA)
    parsed_features['image/human/resized'].set_shape([h, w, 3])
    image = tf.cast(parsed_features['image/human/resized'], tf.uint8)
    bgr_avg = tf.constant(127.5)
    parsed_features['image/human/resized_and_subtract_mean'] = (parsed_features['image/human/resized'] - bgr_avg) * tf.constant(0.0078125)
    
    
    # return image, \
    #        parsed_features['image/human/resized_and_subtract_mean'], \
    #        parsed_features['heatmap'], \
    #        parsed_features['PAF'], \
    #        parsed_features['image/filename']
    return parsed_features['image/human/resized_and_subtract_mean'], \
            {0:parsed_features['heatmap'], 
            1:parsed_features['PAF']}


def data_pipeline(tf_record_path, params={}, batch_size=64, num_parallel_calls=8):
    #preprocess.py -preprocess()
    preprocess = Preprocess()
    tfd = tf.data
    dataset = tfd.TFRecordDataset(tf_record_path)
    dataset = dataset.map(
        _parse_function,
        num_parallel_calls=num_parallel_calls
    )
    if params['do_data_augmentation']:
        dataset = dataset.map(
            preprocess.data_augmentation,
            num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.map(
        preprocess.pyfn_interface_input,
        num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.map(
        lambda img, source, h, w, bx1, bx2, by1, by2, kx, ky, kv, nkp: tuple(tf.py_func(
            preprocess.head_encoder,
            [img, source, h, w, bx1, bx2, by1, by2, kx, ky, kv, nkp],
            [tf.uint8, tf.string, tf.float32, tf.float32, tf.float32, tf.int64])
        ),
        num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.map(
        preprocess.pyfn_interface_output,
        num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.map(
        partial(_preprocess_function, params=params),
        num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.batch(batch_size).prefetch(2 * batch_size)
    iterator = dataset.make_one_shot_iterator()
    data, paf_cla, paf_reg3 = iterator.get_next()
    return data, [paf_cla, paf_reg3]
    # return dataset


def main(_):
    np.set_printoptions(threshold=np.inf)
    task_graph = tf.Graph()
    with task_graph.as_default():
<<<<<<< HEAD
        data, hm_kps, hm_limbs, kps_shape, n_kps = data_pipeline([FLAGS.dataset_path],params={'do_data_augmentation': True, 'dataset_split_num':1},batch_size=FLAGS.batch_size)
=======
        data, paf_cla, paf_reg3 = data_pipeline([FLAGS.dataset_path],params={'do_data_augmentation': False, 'dataset_split_num':1},batch_size=FLAGS.batch_size)
>>>>>>> shufflenetv2_d2s_mulgpu
        sess = tf.Session()
        while True:
            try:
                dataset = sess.run([data, paf_cla, paf_reg3])
            except tf.errors.OutOfRangeError:
                break

            # image = cv2.cvtColor(r_image[0], cv2.COLOR_RGB2BGR)
            # cv2.imshow('image', image)
            # print(prid_kp_np.shape)
            # print(x)
            # prid_kp_np = prid_kp_np[0]
            # prid_kp_np = (prid_kp_np*255).astype("uint8")
            # cv2.imshow('prid_kp_np', prid_kp_np[6])

            # for i in range(18):
            #     print(i)
            #     cv2.imshow('kp', (prid_kp_np[i]* 255).astype("uint8"))
            #     if cv2.waitKey(0) & 0xFF == ord('q'):
            #         break
            print(dataset)
        #     print(new_kps)
        #     con_hm = np.zeros_like(prid_kp_np[0])
        #     for i in range(18):
        #         con_hm += prid_kp_np[i]
        #         mask = con_hm > 1
        #         con_hm[mask] = 1

        #     con_hm = (con_hm * 255).astype("uint8")
            
        #     print(kps_shape_np)

        #     cv2.imshow('con_hm', con_hm)
            
        #     if cv2.waitKey(0) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()