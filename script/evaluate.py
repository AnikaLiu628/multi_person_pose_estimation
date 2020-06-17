import os
import logging
from functools import partial

import math
import numpy as np
import tensorflow as tf

from input_pipeline import Pipeline
from losses import MSMSE


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string(
    'validationset_path',
    '../data/coco_keypoints_val.record-00000-of-00001',
    'Validation data'
)
flags.DEFINE_string(
    'output_model_path',
    '../models/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1',
    'Path of output emotion model'
)
flags.DEFINE_string(
    'model_type',
    'MobilePaf',
    'Model architecture in [MobilePose, FPMobilePose, SEMobilePose, sppe, MobilePifPaf, MobilePaf]'
)
flags.DEFINE_string(
    'backbone',
    'mobilenet_v1',
    'Model backbone in [mobilenet_v1, mobilenet_v2, mobilenet_v3, SEResnet]'
)
flags.DEFINE_string(
    'loss_fn',
    'MSE',
    'Loss function in [MSE, softmax, center, focal, inv_focal, arcface]'
)
flags.DEFINE_float(
    'layer_depth_multiplier',
    1.0,
    'Depth multiplier of mobilenetv1 architecture'
)
flags.DEFINE_integer(
    'number_keypoints',
    18,
    'Number of keypoints in [17, 12]'
)
flags.DEFINE_integer(
    'batch_size',
    16,
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
    40000,
    'Decay steps of learning rate'
)
flags.DEFINE_float(
    'decay_factor',
    0.1,
    'Decay factor of learning rate'
)
flags.DEFINE_integer(
    'training_steps',
    80000,
    'Train n steps'
)
flags.DEFINE_integer(
    'validation_interval',
    1000,
    'Evaluate validation loss for every n steps'
)
flags.DEFINE_integer(
    'validation_batch_size',
    128,
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
        # loss = tf.losses.mean_squared_error(labels[0], net_dict['heat_map'])
        paf_intensities_l = labels[0] #ground thruth field map
        paf_regression3_l = labels[1] #ground thruth vector map
        
        paf_intensities_x = net_dict['heat_map'] #len(3) # predict field map (xs, ys, vs)
        paf_regression3_x = net_dict['PAF']#?, 19, 46, 80 # predict vector map
        # print(labels[0].shape)
        paf_shape = [None,18,46,80]
        reg_shape = [None,38,46,80]
       
        hm_loss = tf.losses.mean_squared_error(paf_intensities_l[:, :, :, :], paf_intensities_x[:, :, :-2, :])
        paf_loss = tf.losses.mean_squared_error(paf_regression3_l[:, :, :, :], paf_regression3_x[:, :, :-2, :])

        loss = hm_loss + 5 * paf_loss

    elif loss_fn == 'MSMSE':
        loss = MSMSE(net_dict, labels)
    elif loss_fn == 'softmax':
        raise ValueError(
            'Softmax not yet implemented.'
            'Please select [MSE]'
        )
    elif loss_fn == 'center':
        raise ValueError(
            'Center loss not yet implemented.'
            'Please select [MSE]'
        )
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers)
    elif loss_fn == 'focal':
        raise ValueError(
            'Focal loss not yet implemented.'
            'Please select [MSE]'
        )
    elif loss_fn == 'inv_focal':
        raise ValueError(
            'Inverse focal loss not yet implemented.'
            'Please select [MSE]'
        )
    elif loss_fn == 'arcface':
        raise ValueError(
            'Arcface loss not yet implemented.'
            'Please select [MSE]'
        )
    else:
        raise ValueError(
            'Loss function is not supported.'
            'Please select [MSE]'
        )
    if loss_only:
        return None, loss
    if Optimizer == 'Momentum':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif Optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif Optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif Optimizer == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif Optimizer == 'Nadam':
        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError('''{} optimizer is not supported. 
            Please choose one of ["Momentum", "Adagrad", "Adam", "RMSProp", "Nadam"]''')
    train = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([train]):
        train_ops = tf.group(*update_ops)
    return train_ops, loss


def model_fn(features, labels, mode, params):
    if params['model_arch'] == 'MobilePose':
        from mobilepose import MobilePose
        model_arch = MobilePose
        output = 'heat_map'
        multi_layer_labels = labels[0]
        hm_labels = labels[0][0]
        labels_weight = labels[1]
    elif params['model_arch'] == 'FPMobilePose':
        from fp_mobilepose import FPMobilePose
        model_arch = FPMobilePose
        output = 'heat_map_4'
        multi_layer_labels = labels[0]
        hm_labels = labels[0][3]
        labels_weight = labels[1]
    elif params['model_arch'] == 'SEMobilePose':
        from se_mobilepose import SEMobilePose
        model_arch = SEMobilePose
        output = 'heat_map'
        multi_layer_labels = labels[0]
        hm_labels = labels[0][0]
        labels_weight = labels[1]
    elif params['model_arch'] == 'sppe':
        from sppe import FastPose
        model_arch = FastPose
        output = 'heat_map'
        multi_layer_labels = labels[0]
        hm_labels = labels[0][0]
        labels_weight = labels[1]
    elif params['model_arch'] == 'MobilePifPaf':
        from mobilepifpaf import MobilePifPaf
        model_arch = MobilePifPaf
        output = 'heat_map'
        multi_layer_labels = labels[0]
        hm_labels = labels[0][0]
        labels_weight = labels[0]
    elif params['model_arch'] == 'MobilePaf':
        from mobilepaf import MobilePaf
        model_arch = MobilePaf
        output = 'heat_map'
        multi_layer_labels = labels[0]
        hm_labels = labels[0][0]
        labels_weight = labels[0]
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

    def find_keypoints(heat_map):
        inds = []
        # print(heat_map)
        for k in range(params['number_keypoints']):
            ind = tf.unravel_index(
                tf.argmax(
                    tf.reshape(heat_map[k, :, :], [-1])),
                [24, 40]
            )# 最大值index
            inds.append(tf.cast(ind, tf.float32)) 
        return tf.stack(inds) #concate 在dim=0
    keypoints_pridict = tf.map_fn(find_keypoints,
                                  end_points[output],
                                  back_prop=False)
    keypoints_labels = tf.map_fn(find_keypoints,
                                 hm_labels,
                                 back_prop=False)
    err = tf.losses.mean_squared_error(keypoints_labels, keypoints_pridict, labels_weight)

    kpt_oks_sigmas = tf.constant([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 
                                  0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])

    def computeOks(kps_prid, kps_lab, oks_sigmas, labels_weight):
        # oks = tf.py_func(computeOks ,[keypoints_pridict, 
        #                   keypoints_labels, kpt_oks_sigmas, labels_weight], tf.float64)
        # multi_layer_labels = labels[0]
        # hm_labels = labels[0][0]
        # labels_weight = labels[0]
        print(kps_prid)
        oks = []
        for j in range(kps_prid.shape[0]):
            #(1)
            mask = labels_weight[j][:, 0] > 0
            if sum(mask) > 0:
                h = kps_lab[j][mask][:, 0].max() - kps_lab[j][mask][:, 0].min()
                w = kps_lab[j][mask][:, 1].max() - kps_lab[j][mask][:, 1].min()
                area = h * w
                if area == 0:
                    area = 23 * 40
                for i in range(17):
                    dx = np.square(kps_prid[j][i][0] - kps_lab[j][i][0])
                    dy =  np.square(kps_prid[j][i][1] - kps_lab[j][i][1])
                    sigma = np.square(oks_sigmas[i] * 2) * area
                    e = np.exp((dx+dy) / (sigma) / -2)
                    if labels_weight[j][i][0] > 0:
                        oks.append(e)
        return np.mean(oks)

    ops_pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]

    def computeOps(kps_prid, kps_lab, ops_pairs, labels_weight, angle_limit):
        ops = []
        for j in range(kps_prid.shape[0]):
            for kp1, kp2 in ops_pairs:
                if labels_weight[j][kp1, 0] > 0 and labels_weight[j][kp2, 0] > 0:
                    vec_lab = (kps_lab[j][kp2][1] - kps_lab[j][kp1][1], kps_lab[j][kp2][0] - kps_lab[j][kp1][0])
                    vec_prid = (kps_prid[j][kp2][1] - kps_prid[j][kp1][1], kps_prid[j][kp2][0] - kps_prid[j][kp1][0])
                    norm_lab = np.sqrt(vec_lab[0]**2 + vec_lab[1]**2)
                    norm_prid = np.sqrt(vec_prid[0]**2 + vec_prid[1]**2)
                    cosine = (vec_lab[0]*vec_prid[0]+vec_lab[1]*vec_prid[1]) / ((norm_lab*norm_prid)+0.00000001)
                    theta = math.degrees(math.acos(round(cosine, 2)))
                    if theta > 90:
                        theta = 180 - theta
                    ops.append(np.max((angle_limit-theta), 0)/angle_limit)
        return np.mean(ops)

    angle_limit = 45
    
    oks = tf.py_func(computeOks ,[keypoints_pridict, keypoints_labels, kpt_oks_sigmas, labels_weight], tf.float64)
    ops = tf.py_func(computeOps ,[keypoints_pridict, keypoints_labels, ops_pairs, labels_weight, angle_limit], tf.float64)

    if mode == tf.estimator.ModeKeys.EVAL:
        mean_err, mean_err_op = tf.metrics.mean(err)
        mean_oks, mean_oks_op = tf.metrics.mean(oks)
        mean_ops, mean_ops_op = tf.metrics.mean(ops)
        mean_loss, mean_loss_op = tf.metrics.mean(loss)
        evaluation_hook = tf.train.LoggingTensorHook(
            {'Global Steps': tf.train.get_global_step(),
             'Distance Error': mean_err_op,
             'OKS': mean_oks_op,
             'OPS': mean_ops_op,
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
         'Distance Error': err,
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
    reader = tf.train.NewCheckpointReader(pretrained_model_path + '/mobilenet_v1_1.0_128.ckpt')
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
    saver.restore(session, data_path + '/mobilenet_v1_1.0_128.ckpt')
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
        session_config.gpu_options.allow_growth=True
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
    logging.basicConfig(
        filename='../logs/' + FLAGS.output_model_path.split('/')[-1] + '_eval_newMetric.log',
        level=logging.INFO
    )
    logging.info(
        ('\n validation data number: {} \n').format(
            len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path)))
        )
    )
    tf.app.run()
