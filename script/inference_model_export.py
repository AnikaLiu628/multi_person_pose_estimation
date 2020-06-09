import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string(
    'model',
    '../models/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v14/model.ckpt-102000',
    'CKPT PATH'
)
flags.DEFINE_string(
    'output_graph',
    'MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v14.pb',
    'PB PATH'
)
flags.DEFINE_string(
    'model_type',
    'MobilePifPaf',
    'Model architecture in [MobilePifPaf]'
)
flags.DEFINE_string(
    'backbone',
    'mobilenet_v1',
    'Model backbone in [mobilenet_v1, mobilenet_v2, shufflenet_v2]'
)
flags.DEFINE_string(
    'input_node',
    'image',
    'Node name of input'
)
flags.DEFINE_string(
    'output_nodes',
    # 'pif/transpose,pif/transpose_1,pif/transpose_2,pif/transpose_3,paf/transpose,paf/transpose_1,paf/transpose_2,paf/transpose_3,paf/transpose_4',
    'hm_out,paf_out',
    'Nodes of output, seperated by comma'
)
flags.DEFINE_float(
    'layer_depth_multiplier',
    1.0,
    'Depth multiplier of mobilenetv1 architecture'
)
FLAGS = flags.FLAGS


def save_and_frezze_model(sess,
                          checkpoint_path,
                          input_nodes,
                          output_nodes,
                          pb_path):
    print('======================================')
    print('Saving .ckpt files...')
    saver = tf.train.Saver()
    saver.save(sess, checkpoint_path)
    print('***.ckpt files have been saving to', checkpoint_path)

    print('======================================')
    print('Saving .pbtxt file...')
    graph_name = os.path.basename(pb_path).replace('.pb', '.pbtxt')
    graph_folder = os.path.dirname(pb_path)
    tf.train.write_graph(sess.graph.as_graph_def(),
                         graph_folder,
                         graph_name,
                         True)
    print('***.pbtxt file has been saving to', graph_folder)

    print('======================================')
    print('Saving .pb file...')
    input_graph_path = os.path.join(graph_folder, graph_name)
    input_saver_def_path = ''
    input_binary = False
    input_checkpoint_path = checkpoint_path
    output_node_names = output_nodes
    restore_op_name = 'unused'
    filename_tensor_name = 'unused'
    output_graph_path = pb_path
    clear_devices = True
    initializer_nodes = ''

    freeze_graph(input_graph_path, input_saver_def_path,
                 input_binary, input_checkpoint_path,
                 output_node_names, restore_op_name,
                 filename_tensor_name, output_graph_path,
                 clear_devices, initializer_nodes)
    print('***.pb file has been saving to', output_graph_path)

    print('======================================')
    print('Saving .tflite file...')
    tflite_path = output_graph_path.replace('.pb', '.tflite')
    input_arrays = input_nodes.split(',')
    output_arrays = output_nodes.split(',')

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        output_graph_path, input_arrays, output_arrays)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as tflite_f:
        tflite_f.write(tflite_model)
    print('***.tflite file has been saving to', tflite_path)

    print('======================================')


def optimize_inference_model(frozen_graph_path,
                             optimized_graph_path,
                             input_node_names,
                             output_node_names):
    print('Reading frozen graph...')
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_path, 'rb') as f:
        data2read = f.read()
        input_graph_def.ParseFromString(data2read)

    print('Optimizing frozen graph...')
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_node_names.split(','),  # an array of the input node(s)
        output_node_names.split(','),  # an array of the output nodes
        tf.float32.as_datatype_enum
    )

    print('Saving the optimized graph .pb file...')
    with tf.gfile.FastGFile(optimized_graph_path, 'w') as f:
        f.write(output_graph_def.SerializeToString())


def main(_):
    print('Rebuild graph...')
    if FLAGS.model_type == 'MobilePifPaf':
        from mobilepifpaf import MobilePifPaf
        model_arch = MobilePifPaf
    else:
        print('{} not supported.'.format(FLAGS.model_type))
        return 0

    model = model_arch(backbone=FLAGS.backbone,
                       is_training=False,
                       depth_multiplier=FLAGS.layer_depth_multiplier)

    inputs = tf.placeholder(tf.float32,
                            shape=(None, 368, 432, 3),
                            name=FLAGS.input_node)
    end_points = model.build(inputs)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.model)

    output_path = '/'.join(FLAGS.model.split('/')[:-1]) + \
        '/output_model_{}/'.format(FLAGS.model.split('-')[-1])
    output_files = output_path + 'model.ckpt'
    print('Model exporting...')

    save_and_frezze_model(sess,
                          output_files,
                          FLAGS.input_node,
                          FLAGS.output_nodes,
                          output_path + FLAGS.output_graph)
    print('Exporting finished !')


if __name__ == '__main__':
    tf.app.run()
