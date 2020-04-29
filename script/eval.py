import tensorflow as tf
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from numpy import unravel_index
import cv2
import random
import math

from input_pipeline import Pipeline
import matplotlib.pyplot as plt
# from sklearn.preprocessing import normalize
'''
"keypoints": [
            184,82,2,
            189,69,2,
            169,75,2,
            0,0,0,
            147,78,2,
            212,121,2,
            136,142,2,
            246,183,2,
            130,241,2,
            264,209,2,
            203,234,2,
            250,253,2,
            183,272,2,
            347,223,2,
            287,244,2,
            290,331,2,
            261,354,2
            ]

COCO_train2014_000000113521
keypoints = [
            332, 94, 1, 
            337, 76, 2, 
            325, 75, 1, 
            386, 78, 2,
            0, 0, 0,
            442, 154, 2,
            327, 145, 2,
            437, 273, 2, 
            291, 251, 2,
            350, 295, 2,
            227, 293, 2, 
            411, 335, 2, 
            328, 325, 2, 
            378, 471, 1, 
            216, 439, 2, 
            0, 0, 0,
            0, 0, 0]
'''
top_ = 1

class Preprocess():
    def __init__(self,):
        self.sigma = 2.0
    def head_encoder(self, img, height, width):
        keypoints = [
                    184,82,2,
                    189,69,2,
                    169,75,2,
                    0,0,0,
                    147,78,2,
                    212,121,2,
                    136,142,2,
                    246,183,2,
                    130,241,2,
                    264,209,2,
                    203,234,2,
                    250,253,2,
                    183,272,2,
                    347,223,2,
                    287,244,2,
                    290,331,2,
                    261,354,2]

        # store [x, y, v ...] into kx, ky, kv
        kx, ky, kv = [], [], []
        for i in range(len(keypoints)):
            if i % 3 == 0:
                kx.append(keypoints[i])
            elif i % 3 == 1:
                ky.append(keypoints[i])
            elif i % 3 == 2:
                kv.append(keypoints[i])
        
        w = 640 #320 640
        h = 360 #256 360
        scaling_ratio = 8
        field_w = int(w / scaling_ratio) #16
        field_h = int(h / scaling_ratio) #20
        orig_keypoints_set = []
        kx = np.asarray(kx).astype(np.float64)
        ky = np.asarray(ky).astype(np.float64)
        kv = np.asarray(kv).astype(np.float64)
        kx = (kx * (field_w / width)).astype(np.uint16)
        ky = (ky * (field_h / height)).astype(np.uint16)
        orig_keypoints_set.append([(x, y) if v >=1 else (-1000, -1000) for x, y, v in zip(kx, ky, kv)])
        orig_keypoints_set = np.squeeze(orig_keypoints_set)
        num_img_ppl = len(kx)//17

        new_keypoint_sets = []
        transform = [
                (1, 1), (6, 7), (7, 7), (9, 9), (11, 11), (6, 6), (8, 8), (10, 10), (13, 13),
                (15, 15), (17, 17), (12, 12), (14, 14), (16, 16), (3, 3), (2, 2), (5, 5), (4, 4),
            ]
        new_kp = []
        for each_ppl in range(0, num_img_ppl): 
            for idx1, idx2 in transform:
                j1 = orig_keypoints_set[idx1 + 17*each_ppl-1]
                j2 = orig_keypoints_set[idx2 + 17*each_ppl-1]
                _x = int((j1[0] + j2[0]) / 2)
                _y = int((j1[1] + j2[1]) / 2)
                new_kp.append((_x, _y))
        new_keypoint_sets.append(new_kp)
        heatmap = np.zeros((18, field_h, field_w), dtype=np.float32)
        self.get_heatmap(heatmap, new_keypoint_sets)
        heatmap = np.array(heatmap)
        heatmap = heatmap.astype(np.float32)
        return heatmap

    def get_heatmap(self, heatmap, new_keypoint_sets):
        keypoint = new_keypoint_sets[0]
        idx = 0
        for point_x, point_y in keypoint:
            if point_x <= 0:
                continue
            n_idx = idx % 18
            point = (point_x, point_y)
            self.put_heatmap(heatmap, n_idx, point, self.sigma)
            idx+=1
        
        return heatmap.astype(np.float32)

    def put_heatmap(self, heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))
        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        exp_factor = 1 / 2.0 / sigma / sigma

        #fater version_1
        arr_heatmap = heatmap[plane_idx, y0:y1 , x0:x1 ]
        y_vec = (np.arange(y0, y1) - center_y) ** 2  # y1 included
        x_vec = (np.arange(x0, x1) - center_x) ** 2
        xv, yv = np.meshgrid(x_vec, y_vec)
        arr_sum = exp_factor * (xv + yv)
        arr_exp = np.exp(-arr_sum)
        arr_exp[arr_sum > th] = 0
        heatmap[plane_idx, y0:y1, x0:x1] = np.maximum(arr_heatmap, arr_exp)


def init_models(model_name):
    graph = tf.Graph()
    POSE = '../models/' + model_name + '/output_model_49000/' + model_name + '.pb'
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(POSE, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    return graph

def main(_):
    model_name = 'MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v6'
    graph = init_models(model_name)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            input_tensor = graph.get_tensor_by_name('image:0')
            # 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu:0' MobilenetV2/Conv_1/Relu6:0 shufflenet_v2/conv5/Relu:0
            output_tensor = graph.get_tensor_by_name('shufflenet_v2/conv5/Relu:0')
            class_tensor = graph.get_tensor_by_name('paf/class_out:0')

            width = 640 #256 #640 
            hight = 360 #320 #360

            image = cv2.imread('../data/COCO_train2014_000000000113.jpg')
            height_img = image.shape[0]
            width_img = image.shape[1]
            pp = Preprocess()
            heatmap_gt = pp.head_encoder(image, height_img, width_img)

            # i = 9
            # heatmap_gt = heatmap_gt[i]
            # print(heatmap_gt)
            # hm = np.amax(heatmap_gt)
            
            # hm_loc = np.where(heatmap_gt == hm)
            # listCordinates = list(zip(hm_loc[0], hm_loc[1]))
            # for cord in listCordinates:
            #     print(cord)
            # print('Value ', hm)
            # print('GroundTruth Heatmap')

            image_np = image.copy()
            image_resized = cv2.resize(image,dsize=(width,hight))
            image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image= (image - 127.5) * 0.0078125
            image_np_expanded = np.expand_dims(image, axis = 0)

            suff_feature = sess.run(output_tensor, feed_dict = {input_tensor: image_np_expanded})
            class_feature = sess.run(class_tensor, feed_dict = {output_tensor: suff_feature})
            heatmap = class_feature[0] #18, 20, 16
            kps_num = len(heatmap) #18

            # heatmap = heatmap[i]
            # print(heatmap)
            # hm_l_loc = np.where(heatmap == np.amax(heatmap))
            # l_listCordinates = list(zip(hm_l_loc[0], hm_l_loc[1]))
            # for l_cord in l_listCordinates:
            #     print(l_cord)
            # print('Value ', np.amax(heatmap))
            # print('Output Heatmap')

            heatmap_add_gt = np.zeros_like(heatmap_gt[0])
            print(heatmap.shape)
            print(heatmap_gt.shape)
            heatmap_add = np.zeros_like(heatmap_gt[0])
            '''adjust kps num'''
            # for i in range(18):
            i = 2
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
            if i == 12:
                part_name = 'Right knee'
            if i == 13:
                part_name = 'Right ankle'
            if i == 11:
                part_name = 'Right hip'
            heatmap_add += heatmap[i][:-1]
            # print(heatmap_add)
            # heatmap_add += heatmap[i]
            heatmap_add_gt += heatmap_gt[i]

            # norm = np.linalg.norm(an_array)
            # normal_array = an_array/norm

            ## heatmap_add = (heatmap_add * 255).astype("uint8")
            # heatmap_add_resized = cv2.resize(heatmap_add, dsize=(width,hight))
            # heatmap_add_gt_resized = cv2.resize(heatmap_add_gt, dsize=(width,hight))

            # heatmap_add_resized_color = cv2.applyColorMap(heatmap_add_resized, cv2.COLORMAP_JET)
            # heatmap_add_resized_color = cv2.cvtColor(heatmap_add_resized, cv2.COLOR_GRAY2BGR)
            # heatmap_add_resized_color = cv2.cvtColor(heatmap_add_resized_color, cv2.COLOR_BGR2RGB)

            # heatmap_add_resized_color_weg = cv2.addWeighted(heatmap_add_resized_color, 0.7, image_resized, 0.3, 0)
            P = plt.figure(1)
            plt.imshow(heatmap_add_gt, cmap='viridis')
            plt.title('{} \nGround Truth {} feature'.format(model_name, part_name))
            PP = plt.figure(2)
            plt.imshow(heatmap_add, cmap='viridis')
            plt.title('{} \nPredict {} feature'.format(model_name, part_name))
            plt.show()
            # cv2.imshow('Predhm', heatmap_add_resized_color_weg)
            # cv2.waitKey(0) & 0xFF == ord('q')

if __name__ == '__main__':
    tf.app.run()

