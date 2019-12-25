import os
import logging
from functools import partial

import tensorflow as tf

from input_pipeline import Pipeline
from losses import MSMSE


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string(
    'dataset_path',
    '/datasets/coco/intermediate/coco_mp_keypoints_train.record-00000-of-00001',
    'Training data (separated by comma)'
)
flags.DEFINE_string(
    'validationset_path',
    '/datasets/coco/intermediate/coco_mp_keypoints_val.record-00000-of-00001',
    'Validation data'
)
flags.DEFINE_string(
    'output_model_path',
    '/workspace/projects/multi_person_pose_estimation/models/MPPE_MOBILENET_V1_0.5_360_640_v1',
    'Path of output human pose model'
)
flags.DEFINE_string(
    'pretrained_model_path',
    '/workspace/projects/multi_person_pose_estimation/models/pretrained_models/mobilenet_v1_0.5_320_256/model.ckpt',
    'Path of pretrained model(ckpt)'
)
flags.DEFINE_string(
    'model_type',
    'MobilePifPaf',
    'Model architecture in [MobilePifPaf]'
)
flags.DEFINE_string(
    'backbone',
    'shufflenet_v2',
    'Model backbone in [mobilenet_v1, mobilenet_v2, shufflenet_v2]'
)
flags.DEFINE_string(
    'loss_fn',
    'MSE',
    'Loss function in [MSE, softmax, center, focal, inv_focal, arcface, MSE_OHEM]'
)
flags.DEFINE_float(
    'layer_depth_multiplier',
    1.0,
    'Depth multiplier of mobilenetv1 architecture'
)
flags.DEFINE_integer(
    'number_keypoints',
    17,
    'Number of keypoints in [17, 12]'
)
flags.DEFINE_integer(
    'batch_size',
    32,
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
    1000,
    'Evaluate validation loss for every n steps'
)
flags.DEFINE_integer(
    'validation_batch_size',
    256,
    'Size of batch data'
)
flags.DEFINE_integer(
    'ohem_top_k',
    8,
    'online hard example/keypoint mining choice top k keypoint'
)

FLAGS = flags.FLAGS
pretrained_model_steps = 13109116


class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.input_fn = input_fn

    def after_save(self, session, global_step):
        self.estimator.evaluate(
            self.input_fn, steps=len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path))) // FLAGS.validation_batch_size + 1
        )


def train_op(labels, net_dict, loss_fn, learning_rate, Optimizer, global_step=0, ohem_top_k=8):
    if loss_fn == 'MSE':
        pif_intensities_l = labels[0]
        pif_regression_l = labels[1]
        pif_scale_l = labels[2]
        paf_intensities_l = labels[3]
        paf_regression1_l = labels[4]
        paf_regression2_l = labels[5]

        pif_intensities_x = net_dict['PIF'][0]
        pif_regression_x = net_dict['PIF'][1]
        pif_spreads_x = net_dict['PIF'][2]
        pif_scale_x = net_dict['PIF'][3]
        paf_intensities_x = net_dict['PAF'][0]
        paf_regression1_x = net_dict['PAF'][1]
        paf_regression2_x = net_dict['PAF'][2]
        paf_spreads1_x = net_dict['PAF'][3]
        paf_spreads2_x = net_dict['PAF'][4]

        batch_size = tf.cast(tf.shape(pif_intensities_l)[0], tf.float32)
        pif_shape = [None,17,45,80]
        paf_shape = [None,19,45,80]

        def laplace_loss(x1, x2, logb, t1, t2, weight=None):
            norm = tf.norm(tf.stack([x1, x2]) - tf.stack([t1, t2]), axis=0)
            losses = 0.694 + logb + norm * tf.exp(-logb)
            if weight is not None:
                losses = losses * weight
            return tf.reduce_sum(losses)

        bce_masks = pif_intensities_l[:, :-1] + pif_intensities_l[:, -1:] > 0.5
        bce_masks.set_shape(pif_shape)
        # ce_loss = tf.losses.sigmoid_cross_entropy(
        #     tf.boolean_mask(pif_intensities_l[:, :-1, :, :], bce_masks),
        #     tf.boolean_mask(pif_intensities_x, bce_masks))

        mse_loss = tf.losses.mean_squared_error(
            tf.boolean_mask(pif_intensities_l[:, :-1, :, :], bce_masks),
            tf.boolean_mask(pif_intensities_x[:, :, :-1, :], bce_masks))

        # ce_loss = tf.nn.weighted_cross_entropy_with_logits(
        #     tf.boolean_mask(pif_intensities_l[:, :-1, :, :], bce_masks),
        #     tf.boolean_mask(pif_intensities_x, bce_masks),
        #     pos_weight=10)
        # ce_loss = tf.math.reduce_sum(ce_loss)  / 1000.0 / batch_size

        reg_masks = pif_intensities_l[:, :-1] > 0.5
        reg_masks.set_shape(pif_shape)
        reg_losses = laplace_loss(
            tf.boolean_mask(pif_regression_x[:, :, 0, :-1, :], reg_masks),
            tf.boolean_mask(pif_regression_x[:, :, 1, :-1, :], reg_masks),
            tf.boolean_mask(pif_spreads_x[:, :, :-1, :], reg_masks),
            tf.boolean_mask(pif_regression_l[:, :, 0, :, :], reg_masks),
            tf.boolean_mask(pif_regression_l[:, :, 1, :, :], reg_masks)) \
                / 1000.0 / batch_size

        scale_losses = tf.losses.absolute_difference(
            tf.boolean_mask(pif_scale_l, reg_masks),
            tf.boolean_mask(pif_scale_x[:, :, :-1, :], reg_masks)) / 1000.0

        paf_bce_masks = paf_intensities_l[:, :-1] + paf_intensities_l[:, -1:] > 0.5
        paf_bce_masks.set_shape(paf_shape)
        # paf_ce_loss = tf.losses.sigmoid_cross_entropy(
        #     tf.boolean_mask(paf_intensities_l[:, :-1, :, :], paf_bce_masks),
        #     tf.boolean_mask(paf_intensities_x, paf_bce_masks))

        paf_mse_loss = tf.losses.mean_squared_error(
            tf.boolean_mask(paf_intensities_l[:, :-1, :, :], paf_bce_masks),
            tf.boolean_mask(paf_intensities_x[:, :, :-1, :], paf_bce_masks))

        paf_reg_masks = paf_intensities_l[:, :-1] > 0.5
        paf_reg_masks.set_shape(paf_shape)
        paf_reg1_losses = laplace_loss(
            tf.boolean_mask(paf_regression1_x[:, :, 0, :-1, :], paf_reg_masks),
            tf.boolean_mask(paf_regression1_x[:, :, 1, :-1, :], paf_reg_masks),
            tf.boolean_mask(paf_spreads1_x[:, :, :-1, :], paf_reg_masks),
            tf.boolean_mask(paf_regression1_l[:, :, 0, :, :], paf_reg_masks),
            tf.boolean_mask(paf_regression1_l[:, :, 1, :, :], paf_reg_masks)) \
                / 1000.0 / batch_size
        paf_reg2_losses = laplace_loss(
            tf.boolean_mask(paf_regression2_x[:, :, 0, :-1, :], paf_reg_masks),
            tf.boolean_mask(paf_regression2_x[:, :, 1, :-1, :], paf_reg_masks),
            tf.boolean_mask(paf_spreads2_x[:, :, :-1, :], paf_reg_masks),
            tf.boolean_mask(paf_regression2_l[:, :, 0, :, :], paf_reg_masks),
            tf.boolean_mask(paf_regression2_l[:, :, 1, :, :], paf_reg_masks)) \
                / 1000.0 / batch_size

        loss = 30 * mse_loss + 2 * reg_losses + 2 * scale_losses + \
                50 * paf_mse_loss + 3 * paf_reg1_losses + 3 * paf_reg2_losses

    elif loss_fn =='dice':
        pif_intensities_l = labels[0]
        pif_regression_l = labels[1]
        pif_scale_l = labels[2]
        paf_intensities_l = labels[3]
        paf_regression1_l = labels[4]
        paf_regression2_l = labels[5]

        pif_intensities_x = net_dict['PIF'][0]
        pif_regression_x = net_dict['PIF'][1]
        pif_spreads_x = net_dict['PIF'][2]
        pif_scale_x = net_dict['PIF'][3]
        paf_intensities_x = net_dict['PAF'][0]
        paf_regression1_x = net_dict['PAF'][1]
        paf_regression2_x = net_dict['PAF'][2]
        paf_spreads1_x = net_dict['PAF'][3]
        paf_spreads2_x = net_dict['PAF'][4]

        smooth = 1.0
        intersection = pif_intensities_l[:, :-1] * pif_intensities_x
        loss = (2.0 * tf.reduce_sum(intersection) + smooth) / \
            (tf.reduce_sum(pif_intensities_l[:, :-1]) * tf.reduce_sum(pif_intensities_x) + smooth)
    elif loss_fn =='WCE':
        pif_intensities_l = labels[0]
        pif_regression_l = labels[1]
        pif_scale_l = labels[2]
        paf_intensities_l = labels[3]
        paf_regression1_l = labels[4]
        paf_regression2_l = labels[5]

        pif_intensities_x = net_dict['PIF'][0]
        pif_regression_x = net_dict['PIF'][1]
        pif_spreads_x = net_dict['PIF'][2]
        pif_scale_x = net_dict['PIF'][3]
        paf_intensities_x = net_dict['PAF'][0]
        paf_regression1_x = net_dict['PAF'][1]
        paf_regression2_x = net_dict['PAF'][2]
        paf_spreads1_x = net_dict['PAF'][3]
        paf_spreads2_x = net_dict['PAF'][4]

        batch_size = tf.cast(tf.shape(pif_intensities_l)[0], tf.float32)
        all_num = tf.reduce_prod(tf.cast(tf.shape(pif_intensities_l[:, :-1]), tf.float32))
        pos_num = tf.reduce_sum(pif_intensities_l[:, :-1])
        wce_loss = tf.nn.weighted_cross_entropy_with_logits(
            pif_intensities_l[:, :-1, :, :],
            pif_intensities_x,
            pos_weight=(all_num - pos_num) * pos_num)
        wce_loss = tf.reduce_sum(wce_loss) / (all_num - pos_num) / 1000.0 / batch_size

        loss = wce_loss

    elif loss_fn == 'MSMSE':
        loss = MSMSE(net_dict, labels)
    elif loss_fn == 'MSE_OHEM':
        loss_pre = tf.losses.mean_squared_error(labels[0], net_dict['heat_map'], reduction=tf.losses.Reduction.NONE)
        loss_pre = tf.reduce_mean(loss_pre, (1,2))
        sub_loss, _ = tf.nn.top_k(loss_pre, ohem_top_k, name="ohem_test")
        loss = tf.reduce_mean(sub_loss)
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
    if Optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
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
            Please choose one of ["Momentum", "Adagrad", "Adam", "RMSProp", "Nadam"]'''.format(Optimizer))

    train = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([train]):
        train_ops = tf.group(*update_ops)
    return train_ops, loss


def model_fn(features, labels, mode, params):
    if params['model_arch'] == 'MobilePifPaf':
        from mobilepifpaf import MobilePifPaf
        model_arch = MobilePifPaf
        output = 'outputs'
        multi_head_labels = labels
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

    else:
        model = model_arch(backbone=params['backbone'],
                           is_training=False,
                           depth_multiplier=params['layer_depth_multiplier'],
                           number_keypoints=params['number_keypoints'])

        end_points = model.build(features)

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
        multi_head_labels,
        end_points,
        loss_fn=params['loss_fn'],
        learning_rate=learning_rate,
        Optimizer=params['optimizer'],
        global_step=tf.train.get_global_step(),
        ohem_top_k=params['ohem_top_k']
    )

    if mode == tf.estimator.ModeKeys.EVAL:
        mean_loss, mean_loss_op = tf.metrics.mean(loss)
        evaluation_hook = tf.train.LoggingTensorHook(
            {'Global Steps': tf.train.get_global_step(),
             'Evaluation Loss': mean_loss_op,
             'Batch Size': tf.shape(multi_head_labels[0])[0],
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
         'Training Loss': loss,
         'Learning Rate': learning_rate},
        every_n_iter=100,
        every_n_secs=None,
        at_end=False
    )
    if params['pretrained_model']:
        saver = get_pretrained_model_saver(params['pretrained_model_path'])
        load_fn = partial(load_for_estimator,
                          data_path=params['pretrained_model_path'],
                          saver=saver)
        init_load = tf.train.Scaffold(init_fn=load_fn)
    else:
        init_load = None
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train,
                                      scaffold=init_load,
                                      training_hooks=[training_hook])


def get_pretrained_model_saver(pretrained_model_path):
    reader = tf.train.NewCheckpointReader(pretrained_model_path)
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
    saver.restore(session, data_path)
    # saver.restore(session, data_path + '/model.ckpt-13139116')
    session.graph._unsafe_unfinalize()
    session.run(tf.assign(tf.train.get_global_step(), 0))
    session.graph.finalize()


def LR(initial_learning_rate, global_step, decay_steps, decay_factor):
    return initial_learning_rate * decay_factor ** (global_step // decay_steps)


def main(_):
    task_graph = tf.Graph()
    with task_graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        datasets = FLAGS.dataset_path.split(',')
        model_params = {
            'model_arch': FLAGS.model_type,
            'backbone': FLAGS.backbone,
            'loss_fn': FLAGS.loss_fn,
            'optimizer': FLAGS.optimizer,
            'initial_learning_rate': FLAGS.learning_rate,
            'decay_steps': FLAGS.decay_steps,
            'decay_factor': FLAGS.decay_factor,
            'layer_depth_multiplier': FLAGS.layer_depth_multiplier,
            'number_keypoints': FLAGS.number_keypoints,
            'pretrained_model': FLAGS.pretrained_model,
            'pretrained_model_path': FLAGS.pretrained_model_path,
            'ohem_top_k': FLAGS.ohem_top_k
        }
        pipeline_param = {
            'model_arch': FLAGS.model_type,
            'do_data_augmentation': FLAGS.data_augmentation,
            'loss_fn': FLAGS.loss_fn,
            'number_keypoints': FLAGS.number_keypoints,
            'dataset_split_num': len(datasets),
        }

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        config = (
            tf.estimator
            .RunConfig()
            .replace(
                session_config=session_config,
                save_summary_steps=1000,
                save_checkpoints_secs=None,
                save_checkpoints_steps=FLAGS.validation_interval,
                keep_checkpoint_max=1000,
                log_step_count_steps=1000
            )
        )

        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=FLAGS.output_model_path,
                                       config=config,
                                       params=model_params)

        print(
            ('\n validation data number: {} \n').format(
                len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path)))
            )
        )

        pip = Pipeline()
        model.train(
            input_fn=lambda: pip.data_pipeline(
                datasets,
                params=pipeline_param,
                batch_size=FLAGS.batch_size
            ),
            steps=FLAGS.training_steps,
            saving_listeners=[
                EvalCheckpointSaverListener(
                    model,
                    lambda: pip.eval_data_pipeline(
                        FLAGS.validationset_path,
                        params=pipeline_param,
                        batch_size=FLAGS.validation_batch_size
                    )
                )
            ]
        )
        print('Training Process Finished.')


if __name__ == '__main__':
    if not os.path.exists('../logs'):
        os.makedir('../logs')
    logging.basicConfig(
        filename='../logs/' + FLAGS.output_model_path.split('/')[-1] + '.log',
        level=logging.INFO
    )
    logging.info(
        (
            '--dataset_path={0} \\\n' +
            '--validationset_path={1} \\\n' +
            '--output_model_path={2} \\\n' +
            '--pretrained_model_path={3} \\\n' +
            '--model_type={4} \\\n' +
            '--backbone={5} \\\n' +
            '--loss_fn={6} \\\n' +
            '--layer_depth_multiplier={7} \\\n' +
            '--number_keypoints={8} \\\n' +    
            '--batch_size={9} \\\n' +
            '--pretrained_model={10} \\\n' +
            '--data_augmentation={11} \\\n' +
            '--optimizer={12} \\\n' +
            '--learning_rate={13} \\\n' +
            '--decay_steps={14} \\\n' +
            '--decay_factor={15} \\\n' +
            '--training_steps={16} \\\n' +
            '--validation_interval={17} \\\n' +
            '--validation_batch_size={18} \\\n' +
            '--ohem_top_k={19} \n'
        ).format(
            FLAGS.dataset_path,
            FLAGS.validationset_path,
            FLAGS.output_model_path,
            FLAGS.pretrained_model_path,
            FLAGS.model_type,
            FLAGS.backbone,
            FLAGS.loss_fn,
            FLAGS.layer_depth_multiplier,
            FLAGS.number_keypoints,
            FLAGS.batch_size,
            FLAGS.pretrained_model,
            FLAGS.data_augmentation,
            FLAGS.optimizer,
            FLAGS.learning_rate,
            FLAGS.decay_steps,
            FLAGS.decay_factor,
            FLAGS.training_steps,
            FLAGS.validation_interval,
            FLAGS.validation_batch_size,
            FLAGS.ohem_top_k
        )
    )
    logging.info(
        ('\n validation data number: {} \n').format(
            len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path)))
        )
    )
    tf.app.run()
