import tensorflow as tf
import numpy as np
from numpy import unravel_index
import cv2
import random

from input_pipeline import Pipeline
import matplotlib.pyplot as plt

top_ = 1
weight_value = 1
model_name = 'MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v8'

pairs_name = ['LShoulderElbow_X',     'LShoulderElbow_Y', 
                'LElbowWrist_X',        'LElbowWrist_Y', 
                'RShoulderElbow_X',     'RShoulderElbow_Y', 
                'RElbowWrist_X',        'RElbowWrist_Y', 
                'LHipKnee_X',           'LHipKnee_Y', 
                'LKneeAnkle_X' ,        'LKneeAnkle_Y', 
                'RHipKnee_X',           'RHipKnee_Y', 
                'RKneeAnkle_X',         'RKneeAnkle_Y', 
                'LShoulderRShoulder_X', 'LShoulderRShoulder_Y', 
                'LHipRHip_X',           'LHipRHip_Y']

parts_name = [
                'nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder',
                'LElbow', 'REblow', 'LWrist', 'RWrist', 'LHip', 'HHip', 'LKnee', 
                'RKnee', 'LAnkle', 'RAnkle'

]


def init_models():
    # HD = 'saved_model/frozen_inference_graph.pb'
    graph = tf.Graph()
    POSE = '../models/{}/output_model_11000/{}.pb'.format(model_name, model_name)
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(POSE, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    return graph
def save_paf_img(weight_value, paf, width, hight, image_resized):
    for i in range(20):
        part_name = pairs_name[i]

        paf_w = paf[i] * weight_value
        paf_w_ = (paf_w * 255).astype("uint8")
        paf_resize = cv2.resize(paf_w_, dsize=(width,hight))

        paf_resize_color = cv2.applyColorMap(paf_resize, cv2.COLORMAP_JET)
        paf_resize_color_addimg = cv2.addWeighted(paf_resize_color, 0.5, image_resized, 0.5, 0)
        cv2.imwrite('../figures/{}_{}_paf.png'.format(model_name, part_name), paf_resize_color_addimg)

def save_hm_img(weight_value, heatmap, width, hight, image_resized):
    for i in range(17):
        part_name = parts_name[i]

        heatmap_w = heatmap[i] * weight_value
        heatmap_w_ = (heatmap_w * 255).astype("uint8")
        heatmap_resize = cv2.resize(heatmap_w_, dsize=(width,hight))

        heatmap_resize_color = cv2.applyColorMap(heatmap_resize, cv2.COLORMAP_JET)
        heatmap_resize_color_addimg = cv2.addWeighted(heatmap_resize_color, 0.5, image_resized, 0.5, 0)
        cv2.imwrite('../figures/{}_{}_hm.png'.format(model_name, part_name), heatmap_resize_color_addimg)


def main(_):
    graph = init_models()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            input_tensor = graph.get_tensor_by_name('image:0')
            # output_tensor = graph.get_tensor_by_name('shufflenet_v2/conv5/Relu:0')
            # class_tensor = graph.get_tensor_by_name('paf/class_out:0')
            # regre_tensor = graph.get_tensor_by_name('paf/regression_out:0')
            output_tensor = graph.get_tensor_by_name('feat_concat:0')
            class_tensor = graph.get_tensor_by_name('hm_out:0')
            regre_tensor = graph.get_tensor_by_name('paf_out:0')

            width = 432
            hight = 368
            # width = 640
            # hight = 360

            image = cv2.imread('../data/p2.jpg')
            image_np = image.copy()
            image_resized = cv2.resize(image,dsize=(width,hight))
            image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image= (image - 127.5) * 0.0078125
            image_np_expanded = np.expand_dims(image, axis = 0)

            suff_feature = sess.run(output_tensor, feed_dict = {input_tensor: image_np_expanded})
            class_feature = sess.run(class_tensor, feed_dict = {output_tensor: suff_feature})
            regre_feature = sess.run(regre_tensor, feed_dict = {output_tensor: suff_feature})
            heatmap = np.squeeze(class_feature)
            paf = np.squeeze(regre_feature)
            

            kps_num = len(heatmap)
            pairs_num = len(paf)
            '''point to loc.'''
            # max. value for heatmap
            # heatmap_loc = []
            # for kp in range(kps_num):
            #     one_hm = heatmap[kp]
            #     max_values_one_hm = np.sort(one_hm, axis=None)[::-1][:top_]
            #     for h in range(len(one_hm)):
            #         for w in range(len(one_hm[0])):
            #             for max_value in max_values_one_hm:
            #                 if one_hm[h][w] == max_value:
            #                     h_resize = h * float(hight / 45)
            #                     w_resize = w * float(width / 80)
            #                     heatmap_loc.append((h_resize, w_resize))
            
            # > 0.5
            heatmap_loc = []
            for kp in range(kps_num):
                one_hm = heatmap[kp]
                for h in range(len(one_hm)):
                    for w in range(len(one_hm[0])):
                        if one_hm[h][w] >= 0.5:
                            h_resize = h * float(hight / 45)
                            w_resize = w * float(width / 80)
                            heatmap_loc.append((h_resize, w_resize))

            # for i in range(18):
            # # i = 13
            #     x, y = heatmap_loc[i]
            #     if i == 2:
            #         cv2.circle(image_resized, (int(y), int(x)), 5, (0,255,0), -1)
            #     elif i == 3:
            #         cv2.circle(image_resized, (int(y), int(x)), 5, (0,255,0), -1)
            #     elif i == 4:
            #         cv2.circle(image_resized, (int(y), int(x)), 5, (0,255,0), -1)
            #     elif i == 5:
            #         cv2.circle(image_resized, (int(y), int(x)), 5, (255,255,0), -1)
            #     elif i == 6:
            #         cv2.circle(image_resized, (int(y), int(x)), 5, (255,255,0), -1)
            #     elif i == 7:
            #         cv2.circle(image_resized, (int(y), int(x)), 5, (255,255,0), -1)

            '''resize hm to input size'''
            i = 13
            
            if i == 0:
                part_name = 'nose'
            if i == 1:
                part_name = 'neck'
            if i == 2:
                part_name = 'Left shoulder'
            if i == 3:
                part_name = 'Left elbow' 
            if i == 4:
                part_name = 'Left wrist' #手腕
            if i == 5:
                part_name = 'Right shoulder'
            if i == 6:
                part_name = 'Right elbow'
            if i == 7:
                part_name = 'Right wrist'
            if i == 8:
                part_name = 'Left hip'
            if i == 9:
                part_name = 'Left knee'
            if i == 10:
                part_name = 'Left ankle' 
            if i == 11:
                part_name = 'Right hip'
            if i == 12:
                part_name = 'Right knee'
            if i == 13:
                part_name = 'Right ankle'
            if i == 14:
                part_name = 'Right eye'
            if i == 15:
                part_name = 'Left eye'
            if i == 16:
                part_name = 'Right ear'
            if i == 17:
                part_name = 'Left ear'
            if i >= 18:
                part_name = '---'
                
            save_hm_img(weight_value, heatmap, width, hight, image_resized)
            save_paf_img(weight_value, paf, width, hight, image_resized)
            # heatmap = np.absolute(paf)
            # print(heatmap.shape)
            # for j in range(19):
            #     heatmap_ = heatmap[j] * weight_value
            #     heatmap_2 = (heatmap_ * 255).astype("uint8")
            #     heatmap_resize = cv2.resize(heatmap_2, dsize=(width,hight))
            #     heatmap_resize_color = cv2.applyColorMap(heatmap_resize, cv2.COLORMAP_JET)
            #     heatmap_resize_color_addimg = cv2.addWeighted(heatmap_resize_color, 0.5, image_resized, 0.5, 0)
                
            #     # cv2.imwrite('../figures/{}.png'.format(part_name), heatmap_resize_color_addimg)
            #     cv2.imshow('{} feature'.format(j),heatmap_resize_color_addimg)

            #     cv2.waitKey()
            

if __name__ == '__main__':
    tf.app.run()
