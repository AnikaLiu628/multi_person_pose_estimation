import os
import logging
from functools import partial

import math
import numpy as np
import tensorflow as tf

from scipy.ndimage.filters import maximum_filter
from common import estimate_pose, draw_humans, read_imgfile, compute_iou2oks, find_keypoints
from input_pipeline import Pipeline
from losses import MSMSE


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string(
    'validationset_path',
    '../data/coco_mp_keypoints_dynsigma_scale4_val.record-00000-of-00001',  #../data/coco_mp_keypoints_dynsigma_scale4_val.record-00000-of-00001  #/datasets/t3/data/coco/intermediate/coco_mp_keypoints_dynsigma_scale4_val.record-00000-of-00001
    'Validation data'
)
flags.DEFINE_integer(
    'width',
    999,
    'width (432, 108, or 999 is image size)'
)
flags.DEFINE_integer(
    'height',
    999,
    'height (368, 92, or 999 is image size)'
)
flags.DEFINE_string(
    'output_model_path',
    '../models/EVAL_MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1',
    'Path of output emotion model'
)
flags.DEFINE_string(
    'model_type',
    'MobilePaf_out4',
    'Model architecture in [MobilePose, FPMobilePose, SEMobilePose, sppe, MobilePifPaf, MobilePaf, MobilePaf_out4]'
)
flags.DEFINE_string(
    'backbone',
    'mobilenet_thin_FPN',
    'Model backbone in [mobilenet_v1, mobilenet_v2, mobilenet_v3, SEResnet, mobilenet_thin]'
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
    'number_keypoints',
    17,
    'Number of keypoints in [17, 12]'
)
flags.DEFINE_integer(
    'batch_size',
    8,
    'Size of batch data'
)
flags.DEFINE_boolean(
    'pretrained_model',
    False,
    'Use pretrained model or not'
)
flags.DEFINE_boolean(
    'data_augmentation',
    False,
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
    10000000,
    'Decay steps of learning rate'
)
flags.DEFINE_float(
    'decay_factor',
    0.1,
    'Decay factor of learning rate'
)
flags.DEFINE_integer(
    'training_steps',
    1800000,
    'Train n steps'
)
flags.DEFINE_integer(
    'validation_interval',
    1000,
    'Evaluate validation loss for every n steps'
)
flags.DEFINE_integer(
    'validation_batch_size',
    1,
    'Size of batch data'
)
flags.DEFINE_integer(
    'TopN',
    5,
    'N of Top n accuracy'
)
FLAGS = flags.FLAGS

def train_op(labels, net_dict, loss_fn, learning_rate, Optimizer, loss_only=False, global_step=0):
    if loss_fn == 'MSE':
        hm_l = labels[0] #ground thruth heat map
        paf_l = labels[1] #ground thruth vector map

        hm_x = net_dict['heat_map'] #len(3) # predict heat map (xs, ys, vs)
        paf_x = net_dict['PAF']#?, 19, 46, 80 # predict vector map
        
        # with tf.device('/cpu:0'):
        hm_loss = tf.losses.mean_squared_error(hm_l, hm_x)
        paf_loss = tf.losses.mean_squared_error(paf_l, paf_x)

        loss = hm_loss + paf_loss
        # loss = hm_loss  

    if loss_only:
        return None, loss
    if Optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError('''{} optimizer is not supported. 
            Please choose one of ["Momentum", "Adagrad", "Adam", "RMSProp", "Nadam"]''')
    
    train = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies([train]):
        train_ops = tf.group(*update_ops)

    return train_ops, loss


def model_fn(features, labels, mode, params):
    if params['model_arch'] == 'PAFMobilePose':
        from paf_mobilepose import PAFMobilePose
        model_arch = PAFMobilePose
        output = 'heat_map'
        multi_layer_labels = labels
        hm_labels = labels[0]
        paf_labels = labels[1]
        img_w = labels[2]
        img_h = labels[3]
        kxs = labels[4]
        kys = labels[5]
        kvs = labels[6]
        
        labels_weight = labels

    elif params['model_arch'] == 'MobilePaf_out4':
        from mobilepaf_out4 import MobilePaf
        model_arch = MobilePaf
        output = 'heat_map'
        multi_layer_labels = labels #pipline => [hm, paf, w ,h, x, y, v]
        hm_labels = labels[0]
        paf_labels = labels[1]
        img_w = labels[2]
        img_h = labels[3]
        kxs = labels[4]
        kys = labels[5]
        kvs = labels[6]
        
        labels_weight = labels[1]
    else:
        raise ValueError(
            'Model type of {} is not supported.'
            'Please select [MobilePose] or [FPMobilePose]'.format(params['model_arch'])
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        model = model_arch(backbone=params['backbone'],
                           is_training=True,
                           depth_multiplier=params['layer_depth_multiplier'],
                           number_keypoints=params['number_keypoints'])

        end_points = model.build(features)
        loss_only = False

    else:
        model = model_arch(backbone=params['backbone'],
                           is_training=False,
                           depth_multiplier=params['layer_depth_multiplier'],
                           number_keypoints=params['number_keypoints'])

        end_points = model.build(features)
        loss_only = True

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={output: end_points[output]})

    learning_rate = tf.train.exponential_decay(
        params['initial_learning_rate'],
        tf.train.get_global_step(),
        params['decay_steps'],
        params['decay_factor'],
        staircase=True
    )

    train, loss = train_op(
        multi_layer_labels,
        end_points,
        loss_fn=params['loss_fn'],
        learning_rate=learning_rate,
        Optimizer=params['optimizer'],
        loss_only=loss_only,
        global_step=tf.train.get_global_step()
    )

    NMS_Threshold = 0.8
    kpt_oks_sigmas = tf.constant([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 
                                  0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])

    def _OKS(dt_hm, dt_paf, gt_hm, gt_paf, w, h):
        num_human_dt = []
        oks_app = []
        for i in range(len(dt_hm)):
            dt_heatMat = dt_hm[i]; dt_pafMat = dt_paf[i]; gt_heatMat = gt_hm[i]; gt_pafMat = gt_paf[i]
            if(FLAGS.width == 999 and FLAGS.height == 999):
                image_width, image_height = w[i], h[i]
            else:
                image_width, image_height = FLAGS.width, FLAGS.height
            dt_human_list = estimate_pose(dt_heatMat, dt_pafMat) #post-process ===> find kps and connections
            gt_human_list = estimate_pose(gt_heatMat, gt_pafMat)
            
            dt_kps = find_keypoints(dt_human_list, image_width, image_height)
            gt_kps = find_keypoints(gt_human_list, image_width, image_height)
            if len(gt_kps) == 0:
                num_human_dt.append(0)
            else:
                num_human_dt.append(len(dt_kps) / len(gt_kps))

            gt = 0
            oks_mat, iou_mat = compute_iou2oks(dt_kps, gt_kps, gt)

            oks_app.append(np.mean(oks_mat))
        
        return np.array(np.mean(oks_app), dtype=np.float32), \
               np.array(np.mean(num_human_dt), dtype=np.float32)
    
    # oks, h_dt = tf.py_func(_OKS, [end_points['heat_map'], end_points['PAF'], hm_labels, paf_labels, img_w, img_h], [tf.float32, tf.float32])

    def _OKSgt(dt_hm, dt_paf, w, h, xs, ys, vs):
        num_human_dt = []
        oks_app = []
        for i in range(len(dt_hm)): # batch_size
            image_width, image_height = w[i], h[i]
            if(FLAGS.width == 999 and FLAGS.height == 999):
                resize_w, resize_h = w[i], h[i]
            else:
                resize_w, resize_h = FLAGS.width, FLAGS.height
                
            gt_kps = []
            for x, y, v in zip(xs[i], ys[i], vs[i]):
                gt_kps.append(x * (resize_w/image_width))
                gt_kps.append(y * (resize_h/image_height))
                gt_kps.append(v)
            dt_heatMat = dt_hm[i]; dt_pafMat = dt_paf[i]
            dt_human_list = estimate_pose(dt_heatMat, dt_pafMat) #post-process ===> find kps and connections
            
            dt_kps = find_keypoints(dt_human_list, resize_w, resize_h)
            # print("dt_kps ===> ", dt_kps)
            # print("gt_kps ===> ", gt_kps)
            # if len(gt_kps) == 0:
            #     num_human_dt.append(0)
            # else:
            num_human_dt.append(len(dt_kps) / len(gt_kps))

            gt = 1
            oks_mat, iou_mat = compute_iou2oks(dt_kps, gt_kps, gt)

            oks_app.append(np.mean(oks_mat))
        
        return np.array(np.mean(oks_app), dtype=np.float32), \
               np.array(np.mean(num_human_dt), dtype=np.float32)
    
    oks, h_dt = tf.py_func(_OKSgt, [end_points['heat_map'], end_points['PAF'], img_w, img_h, kxs, kys, kvs], [tf.float32, tf.float32])



    if mode == tf.estimator.ModeKeys.EVAL:
        # mean_err, mean_err_op = tf.metrics.mean(err)
        mean_oks, mean_oks_op = tf.metrics.mean(oks)
        mean_h_dt, mean_h_dt_op = tf.metrics.mean(h_dt)
        # mean_ops, mean_ops_op = tf.metrics.mean(ops)
        mean_loss, mean_loss_op = tf.metrics.mean(loss)
        evaluation_hook = tf.train.LoggingTensorHook(
            {'Global Steps': tf.train.get_global_step(),
            #  'Distance Error': mean_err_op,
             'Human Detected': mean_h_dt_op,
             'OKS': mean_oks_op,
            #  'OPS': mean_ops_op,
             'Evaluation Loss': mean_loss_op,
             'Batch Size': tf.shape(hm_labels)[0],
             'Learning Rate': learning_rate},
            every_n_iter=1,
            every_n_secs=None,
            at_end=False
        )
    
        init_counter = tf.train.Scaffold(
            init_fn=tf.local_variables_initializer
        )
    
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          scaffold=init_counter,
                                          evaluation_hooks=[evaluation_hook])
    
    training_hook = tf.train.LoggingTensorHook(
        {'Global Steps': tf.train.get_global_step(),
        #  'Distance Error': err,
         'Training Loss': loss,
         'Learning Rate': learning_rate},
        every_n_iter=100,
        every_n_secs=None,
        at_end=False
    )
    
    saver = get_pretrained_model_saver(params['pretrained_model_path'])
    load_fn = partial(load_for_estimator,
                      data_path=params['pretrained_model_path'],
                      saver=saver)
    init_load = tf.train.Scaffold(init_fn=load_fn)
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train,
                                      scaffold=init_load,
                                      training_hooks=[training_hook])


def get_pretrained_model_saver(pretrained_model_path):
    reader = tf.train.NewCheckpointReader(pretrained_model_path + '/output_model_1710000/model.ckpt')
    # reader = tf.train.NewCheckpointReader(pretrained_model_path + '/mobilenet_v1_1.0_128.ckpt')
    #reader = tf.train.NewCheckpointReader(pretrained_model_path + '/model.ckpt-13139116')
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    return tf.train.Saver(restore_vars)


def load_for_estimator(scaffold, session, data_path, saver):
    '''Load network weights.
    scaffold: tf.train.Scaffold object
    session: tf.Session()
    data_path: The path to the numpy-serialized network weights
    session: The current TensorFlow session
    '''
    print('Global steps:', session.run(tf.train.get_global_step()))
    if session.run(tf.train.get_global_step()) != 0:
        return
    saver.restore(session, data_path + '/output_model_1710000/model.ckpt')
    # saver.restore(session, data_path + '/mobilenet_v1_1.0_128.ckpt')
    #saver.restore(session, data_path + '/model.ckpt-13139116')
    session.graph._unsafe_unfinalize()
    session.run(tf.assign(tf.train.get_global_step(), 0))
    session.graph.finalize()


def main(_):
    task_graph = tf.Graph()
    with task_graph.as_default():
        #global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

        model_params = {
            'model_arch': FLAGS.model_type,
            'backbone': FLAGS.backbone,
            'loss_fn': FLAGS.loss_fn,
            'optimizer': FLAGS.optimizer,
            'initial_learning_rate': FLAGS.learning_rate,
            'decay_steps': FLAGS.decay_steps,
            'decay_factor': FLAGS.decay_factor,
            'layer_depth_multiplier': FLAGS.layer_depth_multiplier,
            'number_keypoints': FLAGS.number_keypoints
        }
        pipeline_param = {
            'model_arch': FLAGS.model_type,
            'do_data_augmentation': FLAGS.data_augmentation,
            'loss_fn': FLAGS.loss_fn,
            'number_keypoints': FLAGS.number_keypoints
        }

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth=True #True
        config = (
            tf.estimator
            .RunConfig()
            .replace(
                     session_config=session_config,
                     save_summary_steps=1000,
                     save_checkpoints_secs=None,
                     save_checkpoints_steps=FLAGS.validation_interval,
                     keep_checkpoint_max=1000,
                     log_step_count_steps=1000)
        )

        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=FLAGS.output_model_path,
                                       config=config,
                                       params=model_params)

        amount_of_validationset = len(list(
            tf.python_io.tf_record_iterator(FLAGS.validationset_path)
        ))
        print(
            ('\n validation data number: {} \n').format(
                amount_of_validationset
            )
        )
        pip = Pipeline()
        model.evaluate(
            input_fn=lambda: pip.eval_data_pipeline(
                FLAGS.validationset_path,
                params=pipeline_param,
                batch_size=FLAGS.validation_batch_size
            ),
            steps=amount_of_validationset // FLAGS.validation_batch_size + 1,
        )
        print('Evaluation Process Finished.')


if __name__ == '__main__':
    if not os.path.exists('../logs'):
        os.makedir('../logs')
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(
        filename='../logs/' + FLAGS.output_model_path.split('/')[-1] + '_eval_{}_{}_newMetric.log'.format(FLAGS.width, FLAGS.height),
        level=logging.INFO
    )
    logging.info(
        ('\n validation data number: {} \n').format(
            len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path)))
        )
    )
    tf.app.run()
