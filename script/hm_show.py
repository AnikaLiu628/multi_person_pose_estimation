import tensorflow as tf
import numpy as np
import cv2

from input_pipeline import Pipeline
import matplotlib.pyplot as plt

def init_models():
    graph = tf.Graph()
    POSE = '../models/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1/output_model_25000/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1.pb'
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(POSE, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    return graph

def main(_):
    graph = init_models()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            input_tensor = graph.get_tensor_by_name('image:0')
            # output_tensor = graph.get_tensor_by_name('shufflenet_v2/conv5/Relu:0')
            output_tensor = graph.get_tensor_by_name('MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu:0')
            class_tensor = graph.get_tensor_by_name('paf/class_out:0')
            regre_tensor = graph.get_tensor_by_name('paf/regression_out:0')

            width = 640
            hight = 360
            image = cv2.imread('../data/COCO_train2014_000000113521.jpg')
            image_np = image.copy()
            image_resized = cv2.resize(image,dsize=(width,hight))
            image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image= (image - 127.5) * 0.0078125
            image_np_expanded = np.expand_dims(image, axis = 0)

            suff_feature = sess.run(output_tensor, feed_dict = {input_tensor: image_np_expanded})
            class_feature = sess.run(class_tensor, feed_dict = {output_tensor: suff_feature})
            regre_feature = sess.run(regre_tensor, feed_dict = {output_tensor: suff_feature})
            class_feature = np.squeeze(class_feature)
            regre_feature = np.squeeze(regre_feature)

            '''class_feature(1, 18, 46, 80)'''
            cls_fig_num = len(class_feature)
            print(class_feature.shape)
            con_hm = np.zeros_like(class_feature[0])
            for i in range(18):
                max_feature = np.max(class_feature[i])
                print(class_feature[i])
                # print(max_feature)
                for j in range(23):
                    for k in range(40):
                        if class_feature[i][j][k] == max_feature:
                            class_feature[i][j][k] = 1
                        else:
                            class_feature[i][j][k] = 0
                # print(np.array(class_feature).shape)
            for i in range(18):
                con_hm += class_feature[i]
                # print(class_feature[i])
                # mask = con_hm > 1
                # con_hm[mask] = 1
            # image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
            cv2.imshow('image', image_resized)
            con_hm = (con_hm * 255).astype("uint8")
            con_hm = cv2.resize(con_hm,dsize=(width,hight))
            cv2.imshow('con_hm', con_hm)
            cv2.waitKey(0) & 0xFF == ord('q')
            
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
