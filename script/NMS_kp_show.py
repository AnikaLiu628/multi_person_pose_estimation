import tensorflow as tf
import numpy as np
from numpy import unravel_index
import cv2
import random


from input_pipeline import Pipeline
import matplotlib.pyplot as plt

from scipy.ndimage import maximum_filter, gaussian_filter

top_ = 1

def init_models():
    # HD = 'saved_model/frozen_inference_graph.pb'
    graph = tf.Graph()
    POSE = '../models/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v11/output_model_158000/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v11.pb'
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(POSE, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    return graph
def save_img(weight_value, heatmap, width, hight, image_resized):
    for i in range(18):
        if i == 0:
            part_name = 'nose'
        elif i == 1:
            part_name = 'neck'
        elif i == 2:
            part_name = 'Left shoulder'
        elif i == 3:
            part_name = 'Left elbow' 
        elif i == 4:
            part_name = 'Left wrist' #手腕
        elif i == 5:
            part_name = 'Right shoulder'
        elif i == 6:
            part_name = 'Right elbow'
        elif i == 7:
            part_name = 'Right wrist'
        elif i == 8:
            part_name = 'Left hip'
        elif i == 9:
            part_name = 'Left knee'
        elif i == 10:
            part_name = 'Left ankle' 
        elif i == 11:
            part_name = 'Right hip'
        elif i == 12:
            part_name = 'Right knee'
        elif i == 13:
            part_name = 'Right ankle'
        elif i == 14:
            part_name = 'Right eye'
        elif i == 15:
            part_name = 'Left eye'
        elif i == 16:
            part_name = 'Right ear'
        elif i == 17:
            part_name = 'Left ear'
        elif i >= 18:
            part_name = '---'
        heatmap_w = heatmap[i] * weight_value
        heatmap_w_ = (heatmap_w * 255).astype("uint8")
        heatmap_resize = cv2.resize(heatmap_w_, dsize=(width,hight))
        heatmap_resize_color = cv2.applyColorMap(heatmap_resize, cv2.COLORMAP_JET)
        heatmap_resize_color_addimg = cv2.addWeighted(heatmap_resize_color, 0.5, image_resized, 0.5, 0)
        cv2.imwrite('../figures/{}.png'.format(part_name), heatmap_resize_color_addimg)

def non_max_suppression(plain, window_size=3, threshold=0.15):
    under_threshold_indices = plain < threshold
    plain[under_threshold_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))

def main(_):
    graph = init_models()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            input_tensor = graph.get_tensor_by_name('image:0')
            #feat_concat (mobilenet_thin 432, 368)
            output_tensor = graph.get_tensor_by_name('feat_concat:0')
            class_tensor = graph.get_tensor_by_name('hm_out:0')
            regre_tensor = graph.get_tensor_by_name('paf_out:0')

            width = 432
            hight = 368

            image = cv2.imread('../data/apink3.jpg')
            image_np = image.copy()
            image_resized = cv2.resize(image,dsize=(width,hight))
            image_ = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image= (image_ - 127.5) * 0.0078125
            image_np_expanded = np.expand_dims(image, axis = 0)

            suff_feature = sess.run(output_tensor, feed_dict = {input_tensor: image_np_expanded})
            class_feature = sess.run(class_tensor, feed_dict = {output_tensor: suff_feature})
            regre_feature = sess.run(regre_tensor, feed_dict = {output_tensor: suff_feature})
            heatmap = np.squeeze(class_feature)
            paf = np.squeeze(regre_feature)

            # if heatmap.shape[2] == 18:
            #     heatmap = np.rollaxis(heatmap, 2, 0)
            print(heatmap.shape)
            _NMS_Threshold = 0.8
            # extract interesting coordinates using NMS.
            coords = []     # [[coords in plane1], [....], ...]
            for plain in heatmap[:]:
                nms = non_max_suppression(plain, 5, _NMS_Threshold)
                coords.append(np.where(nms >= _NMS_Threshold))
            a= 0
            # print('plane1_allx:', len(coords[0][0])) #plane1_ally: coords[0][1]
            for k in range(len(coords)):
                for i in range(len(coords[k][0])):
                    # cv2.circle(image_resized, (coords[k][1][i]*8, coords[k][0][i]*8), 5, (0,255,0), -1) #(coords[0][j][i], coords[0][j][i])
                    a += 1
            parts = ['nose_white', 'right eye_white', 'left eye_white', 'right ear_white', 'left ear_white', 
                    'right shoulder_purple', 'left shoulder_blue', 'right elbow_purple', 'left elbow_blue', 
                    'right wrist_purple', 'left wrist_blue', 'right hip_yellow', 'left hip_red', 'right knee_yellow',
                    'left knee_red', 'right ankle_yellow', 'left ankle_red']
            for p in range(17):
                print(parts[p], len(coords[p][1]))
                for i in range(len(coords[p][0])):
                    if any(p == c for c in [5, 7, 9]):
                        point_color = (255, 0, 255) # purple
                    elif any(p == c for c in [11, 13, 15]):
                        point_color = (0, 255, 255) #yellow
                    elif any(p == c for c in [6, 8, 10]):
                        point_color = (255, 0, 0) #blue
                    elif any(p == c for c in [12, 14, 16]):
                        point_color = (0, 0, 255) #red
                    else:
                        point_color = (255, 255, 255) #white
                    cv2.circle(image_resized, (coords[p][1][i]*8, coords[p][0][i]*8), 4, point_color, -1)
                cv2.imshow('img',image_resized)
                cv2.waitKey()
            # print('plane1_allx_ally:', len(coords[0]))
            # print(coords)
            # print(len(coords))

if __name__ == '__main__':
    tf.app.run()
