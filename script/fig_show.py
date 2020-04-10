import tensorflow as tf
import numpy as np
from numpy import unravel_index
import cv2
import random

from input_pipeline import Pipeline
import matplotlib.pyplot as plt

top_ = 1

def init_models():
    # HD = 'saved_model/frozen_inference_graph.pb'
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
            heatmap = np.squeeze(class_feature)

            kps_num = len(heatmap)

            heatmap_loc = []
            for kp in range(kps_num):
                #取出kps最大值, scaling
                one_hm = heatmap[kp]
                max_values_one_hm = np.sort(one_hm, axis=None)[::-1][:top_]
                for h in range(len(one_hm)):
                    for w in range(len(one_hm[0])):
                        for max_value in max_values_one_hm:
                            if one_hm[h][w] == max_value:
                                h_resize = h * float(hight / 23)
                                w_resize = w * float(width / 40)
                                heatmap_loc.append((h_resize, w_resize))

            for i in range(18):
                x, y = heatmap_loc[i]
                cv2.circle(image_resized, (int(y), int(x)), 5, (0,255,0), -1)

            cv2.imshow('img',image_resized)
            cv2.waitKey()
            
if __name__ == '__main__':
    tf.app.run()
