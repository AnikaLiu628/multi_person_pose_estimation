from time import time

import cv2
import numpy as np
import scipy
import tensorflow as tf
import math
import time
import numba as nb
from numba import jit

class Preprocess():
    def __init__(self, kernel_size=4):
        self.skeleton = [
            (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14), (2, 3),
            (3, 4), (4, 5), (3, 17), (2, 6), (6, 7), (7, 8), (6, 18), (2, 1),
            (1, 15), (1, 16), (15, 17), (16, 18),
        ]

        self.kernel_size = kernel_size
        self.min_size = 3
        self.sigma = 8.0

    def pyfn_interface_input(self, parsed_features):
        return parsed_features['image/decoded'], parsed_features['image/filename'], \
            parsed_features['image/height'], parsed_features['image/width'], \
            parsed_features['image/human/bbox/xmin'], parsed_features['image/human/bbox/xmax'], \
            parsed_features['image/human/bbox/ymin'], parsed_features['image/human/bbox/ymax'], \
            parsed_features['image/human/keypoints/x'], parsed_features['image/human/keypoints/y'], \
            parsed_features['image/human/keypoints/v'], parsed_features['image/human/num_keypoints']

    def pyfn_interface_output(self, img, source, paf_intensities, paf_fields_reg3, keypoint_sets, n_kps):
        parsed_features = {
            'image/decoded': img, 
            'image/filename': source,
            'image/paf/intensities': paf_intensities,
            'image/paf/fields_reg3': paf_fields_reg3, 
            'image/keypoint_sets': keypoint_sets,
            'image/kps_shape': n_kps
        }
        return parsed_features

    def head_encoder(self, img, source, height, width, bbx1, bbx2, bby1, bby2, kx, ky, kv, nkp):
        w = 640
        h = 360
        scaling_ratio = 16
        width_ratio = width / w 
        height_ratio = height / h
        padding = 10
        sigma = 8
        field_h = int(h / scaling_ratio) #
        field_w = int(w / scaling_ratio)
        num_keypoints = 17
        paf_n_fields = 19
        keypoint_sets=[]
        num_ppl = len(kx) // 17
        kx = (kx.astype(np.float64) / (width_ratio * scaling_ratio)).astype(np.uint16)
        ky = (ky.astype(np.float64) / (height_ratio * scaling_ratio)).astype(np.uint16)
        keypoint_sets.append([(x, y) if v >=1 else (-1000, -1000) for x, y, v in zip(kx, ky, kv)])
        
        bx1 = np.reshape(bbx1, [-1, 1])
        bx1 = bx1.astype(np.float64) / (width_ratio * scaling_ratio)
        bx2 = np.reshape(bbx2, [-1, 1])
        bx2 = bx2.astype(np.float64) / (width_ratio * scaling_ratio)
        by1 = np.reshape(bby1, [-1, 1])
        by1 = by1.astype(np.float64) / (width_ratio * scaling_ratio)
        by2 = np.reshape(bby2, [-1, 1])
        by2 = by2.astype(np.float64) / (width_ratio * scaling_ratio)
        bbox = np.concatenate([bx1, bx2, by1, by2], axis=1).astype(np.int32)#

        bg = np.ones((1, h, w), dtype=np.float32)
        for xxyy in bbox:
            bg[:, xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]] = 0
        bg = scipy.ndimage.binary_erosion(bg, iterations=2, border_value=1)
        
        kp_tr_st = time.time()
        new_keypoint_sets = []
        transform = [
            (1, 1), (6, 7), (7, 7), (9, 9), (11, 11), (6, 6), (8, 8), (10, 10), (13, 13),
            (15, 15), (17, 17), (12, 12), (14, 14), (16, 16), (3, 3), (2, 2), (5, 5), (4, 4),
        ]
        num_img_ppl = len(kx)//17
        # print('kps shape',np.array(keypoint_sets).shape)
        prev_kp = keypoint_sets[0]
        new_kp = []
        for each_ppl in range(0, num_img_ppl): 
            for idx1, idx2 in transform:
                j1 = prev_kp[idx1 + 17*each_ppl-1]
                j2 = prev_kp[idx2 + 17*each_ppl-1]
                _x = int((j1[0] + j2[0]) / 2)
                _y = int((j1[1] + j2[1]) / 2)
                new_kp.append((_x, _y))
        new_keypoint_sets.append(new_kp)
        # print('kps_tr time cost: ==>', time.time()-kp_tr_st)
        heatmap = np.zeros((paf_n_fields-1, field_h, field_w), dtype=np.float32)
        paf_fields_reg3 = np.zeros((paf_n_fields * 2, field_h, field_w), dtype=np.float32)

        self.get_heatmap(heatmap, new_keypoint_sets)
        self.get_vectormap(heatmap, paf_fields_reg3, new_keypoint_sets, num_img_ppl)
        
        heatmap = np.array(heatmap)
        heatmap = heatmap.astype(np.float32)
        paf_fields_reg3 = np.array(paf_fields_reg3)
        paf_fields_reg3 = paf_fields_reg3.astype(np.float32)
        # print(paf_fields_reg3.shape)
        new_keypoint_sets = np.array(new_keypoint_sets)
        new_keypoint_sets = new_keypoint_sets.astype(np.float32)
        
        k_shape = len(new_keypoint_sets[0])
        keypoint_sets = np.array(keypoint_sets)
        keypoint_sets = keypoint_sets.astype(np.float32)
        return img, source, heatmap, paf_fields_reg3, new_keypoint_sets, k_shape
           
    def get_heatmap(self, heatmap, new_keypoint_sets):
        # hm_st = time.time()
        keypoint = new_keypoint_sets[0]
        # for keypoints in new_keypoint_sets:
        #     for idx, point in keypoints:
        #         if point[1] <= 0:
        #             continue
        #         n_idx = idx % 18
                
        #         self.put_heatmap(heatmap, n_idx, point, self.sigma)
        idx = 0
        for point_x, point_y in keypoint:
            # print('idx', idx)
            if point_x <= 0:
                continue
            n_idx = idx % 18
            point = (point_x, point_y)
            self.put_heatmap(heatmap, n_idx, point, self.sigma)
            idx+=1
        
        # print('get_heatmap: ', time.time() - hm_st)
        
        return heatmap.astype(np.float32)

    def put_heatmap(self, heatmap, plane_idx, center, sigma):
        # print('heatmap:',heatmap.shape)
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
        # print(heatmap)
        #faster version_2
        # kp_heat_proj = np.zeros_like(heatmap[plane_idx])
        # kp_width = x1 - x0
        # kp_height = y1 - y0
        # kp_heat = np.zeros((kp_height, kp_width))
        # print(kp_heat.shape)
        # th = 4.6052
        # # kp_height, kp_width = kp_heat.shape

        # x0 = int(np.clip(center_x - kp_width / 2, 0, width))
        # y0 = int(np.clip(center_y - kp_height / 2, 0, height))

        # x1 = int(np.clip(center_x + kp_width / 2, 0, width))
        # y1 = int(np.clip(center_y + kp_height / 2, 0, height))

        # #print([y0,y1,x0,x1])
        # #print([y0-center_y+kp_height//2,y1-center_y+kp_height//2,x0-center_x+kp_height//2,x1-center_x+kp_height//2])

        # kp_heat_proj[y0:y1,x0:x1] = kp_heat[y0-center_y+kp_height//2:y1-center_y+kp_height//2,x0-center_x+kp_height//2:x1-center_x+kp_height//2]

        # kp_heat_proj[y0:y1,x0:x1] = np.max(np.dstack([kp_heat_proj[y0:y1,x0:x1],heatmap[plane_idx][y0:y1,x0:x1]]),axis=-1)
        # kp_heat_proj[kp_heat_proj>1.0] = 1.0


        # heatmap[plane_idx][y0:y1, x0:x1] = kp_heat_proj[y0:y1,x0:x1]

        #original version
        # for y in range(y0, y1):
        #     for x in range(x0, x1):
        #         d = (x - center_x) **2 + (y - center_y) **2
        #         exp = d / 2.0 / sigma / sigma
        #         if exp > th:
        #             continue
        #         heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
        #         heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def get_vectormap(self, heatmap, paf_fields_reg3, new_keypoint_sets, num_img_ppl):
        #new_kp_sets: 1, 19, 2
        # vm_st = time.time()
        keypoint = new_keypoint_sets[0]
        # for keypoint in new_keypoint_sets:
        for plane_idx, (j_idx1, j_idx2) in enumerate(self.skeleton):
            for each_ppl in range(0, num_img_ppl):
                center_from = keypoint[j_idx1 + 18*each_ppl-1]
                center_to = keypoint[j_idx2 + 18*each_ppl-1]
                if center_from[1] <= 0 or center_to [1] <= 0:
                    continue
                
                nidx = (plane_idx + 18 * each_ppl) % 18
                self.put_vectormap(nidx, center_from, center_to, heatmap, paf_fields_reg3)
    
                
        nonzero = np.nonzero(heatmap)
        # heatmap = np.array(heatmap)
        #19, 61, 96
        for p, y, x in zip(nonzero[0], nonzero[1], nonzero[2]):
            if heatmap[p][y][x] <= 0:
                continue
            paf_fields_reg3[p*2+0][y][x] /= heatmap[p][y][x] #
            paf_fields_reg3[p*2+1][y][x] /= heatmap[p][y][x]
        paf_fields_reg3 = paf_fields_reg3.transpose((1, 2, 0))
        # print('get_vectmap:===>', time.time() - vm_st)
        # print(paf_fields_reg3)

    def put_vectormap(self, plane_idx, center_from, center_to, heatmap, paf_fields_reg3, threshold=8):
        _, height, width = paf_fields_reg3.shape[:3]
        vec_x = center_to[0] - center_from[0] #lenth of vector x
        vec_y = center_to[1] - center_from[1] # lenth of vector y

        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        #kp1 to kp2距離
        norm = math.sqrt(vec_x ** 2 + vec_y **2)
        if norm == 0:
            return

        vec_x /= norm #x向量
        vec_y /= norm #y向量

        #faster version
        x = np.arange(min_x, max_x) #p點所有x值
        y = np.arange(min_y, max_y) #p點所有y值

        bec_x = x - center_from[0] #px[]減去x1
        bec_y = y - center_from[1] #py[]減去y1

        # print(len(x))
        # print(len(y))
        # print('bec_x: ', bec_x, bec_x.shape)
        # print('bec_y: ', bec_y, bec_y.shape)
        # print('bec_x: ',bec_x.shape)
        # print(vec_y)
        # print('bec_y: ',bec_y.shape)
        # # print(vec_x.shape)
        dist = np.abs(bec_x * vec_y - bec_y[:,np.newaxis] * vec_x) 

        # print(dist)
        # print(dist.shape)
        
        # print(threshold)

        # if np.any(dist) <= threshold:
        paf_fields_reg3[plane_idx*2+0][min_y:max_y,min_x:max_x][dist<=threshold] = vec_x
        paf_fields_reg3[plane_idx*2+1][min_y:max_y,min_x:max_x][dist<=threshold] = vec_y
        # print('paf_in: ', paf_fields_reg3.shape)
        # print('paf_fields_reg3: ', paf_fields_reg3.shape)
        # print(a)
        #original version
        # for y in range(min_y, max_y):
        #     for x in range(min_x, max_x):
        #         bec_x = x - center_from[0] #vector
        #         bec_y = y - center_from[1]
        #         dist = abs(bec_x * vec_y - bec_y * vec_x)

        #         if dist > threshold:
        #             continue
        #         # print('brf_heatmap[plane_idx][y][x]: ', heatmap[plane_idx][y][x])
        #         # print('shape: ', heatmap.shape)
        #         # heatmap[plane_idx][y][x] += 1
        #         # print('aft_heatmap[plane_idx][y][x]: ', heatmap[plane_idx][y][x])
        #         paf_fields_reg3[plane_idx*2+0][y][x] = vec_x
        #         paf_fields_reg3[plane_idx*2+1][y][x] = vec_y

