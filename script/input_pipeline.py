import math
from functools import partial

import numpy as np
import tensorflow as tf

from preprocesses import Preprocess


class Pipeline():
    def __init__(self, ):
        pass

    def _parse_function(self, example_proto):
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
                    'image/human/keypoints/v': tf.VarLenFeature(tf.int64),
                    'image/human/keypoints/feature':tf.VarLenFeature(tf.int64),
                    'image/human/heatmap':tf.VarLenFeature(tf.float32),
                    'image/human/PAF':tf.VarLenFeature(tf.float32), 
                    'image/human/keypoints/feature/shape':tf.FixedLenFeature((3, ), tf.int64),
                    'image/human/heatmap/shape':tf.FixedLenFeature((3, ), tf.int64),
                    'image/human/PAF/shape':tf.FixedLenFeature((3, ), tf.int64)
                    }
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

        kps_feat = tf.sparse_tensor_to_dense(parsed_features['image/human/keypoints/feature'], default_value=0)
        parsed_features['image/human/keypoints/feature'] = kps_feat
        hm = tf.sparse_tensor_to_dense(parsed_features['image/human/heatmap'], default_value=0.0)
        parsed_features['image/human/heatmap'] = hm
        paf = tf.sparse_tensor_to_dense(parsed_features['image/human/PAF'], default_value=0.0)
        parsed_features['image/human/PAF'] = paf
        
        parsed_features['image/human/keypoints/feature'] = tf.reshape(
            parsed_features['image/human/keypoints/feature'], 
            parsed_features['image/human/keypoints/feature/shape'])
        parsed_features['image/human/heatmap'] = tf.reshape(
            parsed_features['image/human/heatmap'], 
            parsed_features['image/human/heatmap/shape'])
        parsed_features['image/human/PAF'] = tf.reshape(
            parsed_features['image/human/PAF'], 
            parsed_features['image/human/PAF/shape'])
        
        return parsed_features

    def _preprocess_function(self, parsed_features, params={}):
        w = 432
        h = 368
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
        
        parsed_features['image/human/keypoints/feature'] = tf.transpose(parsed_features['image/human/keypoints/feature'], [2, 0, 1])
        parsed_features['image/human/heatmap'] = tf.transpose(parsed_features['image/human/heatmap'], [2, 0, 1])
        parsed_features['image/human/PAF'] = tf.transpose(parsed_features['image/human/PAF'], [2, 0, 1])

        return parsed_features['image/human/resized_and_subtract_mean'], \
                parsed_features['image/human/keypoints/feature'], \
                parsed_features['image/human/heatmap'], \
                parsed_features['image/human/PAF']
        # return parsed_features['image/human/resized_and_subtract_mean'], \
        #        {0:parsed_features['image/human/heatmap'], 
        #        1:parsed_features['image/human/PAF']}

    def data_pipeline(self, tf_record_path, params={}, batch_size=64, num_parallel_calls=2):
        preprocess = Preprocess()
        tfd = tf.data
        dataset = tfd.Dataset.from_tensor_slices(tf_record_path)
        dataset = dataset.interleave(
            lambda x: tfd.TFRecordDataset(x).repeat().shuffle(buffer_size=300),
            cycle_length=params['dataset_split_num']
        )
        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=num_parallel_calls
        )
        if params['do_data_augmentation']:
            dataset = dataset.map(
                preprocess.data_augmentation,
                num_parallel_calls=num_parallel_calls
            )
        # dataset = dataset.map(
        #     preprocess.pyfn_interface_input,
        #     num_parallel_calls=num_parallel_calls
        # )
        # dataset = dataset.map(
        #     lambda img, source, h, w, bx1, bx2, by1, by2, kx, ky, kv, nkp: tuple(tf.py_func(
        #         preprocess.head_encoder,
        #         [img, source, h, w, bx1, bx2, by1, by2, kx, ky, kv, nkp],
        #         [tf.uint8, tf.string, tf.float32, tf.float32, tf.float32, tf.int64])
        #     ),
        #     num_parallel_calls=num_parallel_calls
        # )
        # dataset = dataset.map(
        #     preprocess.pyfn_interface_output,
        #     num_parallel_calls=num_parallel_calls
        # )
        dataset = dataset.map(
            partial(self._preprocess_function, params=params),
            num_parallel_calls=num_parallel_calls
        )
        dataset = dataset.batch(batch_size).prefetch(2 * batch_size)
        iterator = dataset.make_one_shot_iterator()
        img, kps_f, hm, paf = iterator.get_next()
        return img, [hm, paf]
        # return dataset



    def eval_data_pipeline(self, tf_record_path, params={}, batch_size=64, num_parallel_calls=2):
        preprocess = Preprocess()
        tfd = tf.data
        dataset = tfd.TFRecordDataset(tf_record_path)
        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=num_parallel_calls
        )

        # dataset = dataset.map(
        #     preprocess.pyfn_interface_input,
        #     num_parallel_calls=num_parallel_calls
        # )
        # dataset = dataset.map(
        #     lambda img, source, h, w, bx1, bx2, by1, by2, kx, ky, kv, nkp: tuple(tf.py_func(
        #         preprocess.head_encoder,
        #         [img, source, h, w, bx1, bx2, by1, by2, kx, ky, kv, nkp],
        #         [tf.uint8, tf.string, tf.float32, tf.float32, tf.float32, tf.int64])
        #     ),
        #     num_parallel_calls=num_parallel_calls
        # )
        # dataset = dataset.map(
        #     preprocess.pyfn_interface_output,
        #     num_parallel_calls=num_parallel_calls
        # )
        dataset = dataset.map(
            partial(self._preprocess_function, params=params),
            num_parallel_calls=num_parallel_calls
        )
        dataset = dataset.batch(batch_size).prefetch(2 * batch_size)
        iterator = dataset.make_one_shot_iterator()
        img, kps_f, hm, paf = iterator.get_next()
        return img, [hm, paf]
        # return dataset
