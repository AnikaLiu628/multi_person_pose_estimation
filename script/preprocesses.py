import cv2
import numpy as np
import scipy
import tensorflow as tf
import math
import time

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

    def pyfn_interface_output(self, img, source, heat_map, paf, keypoint_sets, n_kps):
        parsed_features = {
            'image/decoded': img, 
            'image/filename': source,
            'heatmap': heat_map,
            'PAF': paf, 
            'image/keypoint_sets': keypoint_sets,
            'image/kps_shape': n_kps
        }
        return parsed_features

    def head_encoder(self, img, source, height, width, bbx1, bbx2, bby1, bby2, kx, ky, kv, nkp):
        w = 640
        h = 360
        scaling_ratio = 4
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

        heatmap = np.zeros((paf_n_fields-1, field_h, field_w), dtype=np.float32)
        paf = np.zeros((paf_n_fields * 2, field_h, field_w), dtype=np.float32)

        self.get_heatmap(heatmap, new_keypoint_sets)
        self.get_vectormap(heatmap, paf, new_keypoint_sets, num_img_ppl)
        
        heatmap = np.array(heatmap)
        heatmap = heatmap.astype(np.float32)
        paf = np.array(paf)
        paf = paf.astype(np.float32)
        new_keypoint_sets = np.array(new_keypoint_sets)
        new_keypoint_sets = new_keypoint_sets.astype(np.float32)
        
        k_shape = len(new_keypoint_sets[0])
        keypoint_sets = np.array(keypoint_sets)
        keypoint_sets = keypoint_sets.astype(np.float32)
        return img, source, heatmap, paf, new_keypoint_sets, k_shape
           
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

        arr_heatmap = heatmap[plane_idx, y0:y1 , x0:x1 ]
        y_vec = (np.arange(y0, y1) - center_y) ** 2  # y1 included
        x_vec = (np.arange(x0, x1) - center_x) ** 2
        xv, yv = np.meshgrid(x_vec, y_vec)
        arr_sum = exp_factor * (xv + yv)
        arr_exp = np.exp(-arr_sum)
        arr_exp[arr_sum > th] = 0
        heatmap[plane_idx, y0:y1, x0:x1] = np.maximum(arr_heatmap, arr_exp)
        
    def get_vectormap(self, heatmap, paf, new_keypoint_sets, num_img_ppl):
        
        keypoint = new_keypoint_sets[0]
        for plane_idx, (j_idx1, j_idx2) in enumerate(self.skeleton):
            for each_ppl in range(0, num_img_ppl):
                center_from = keypoint[j_idx1 + 18*each_ppl-1]
                center_to = keypoint[j_idx2 + 18*each_ppl-1]
                if center_from[1] <= 0 or center_to [1] <= 0:
                    continue
                
                nidx = (plane_idx + 18 * each_ppl) % 18
                self.put_vectormap(nidx, center_from, center_to, heatmap, paf)
    
                
        nonzero = np.nonzero(heatmap)
        for p, y, x in zip(nonzero[0], nonzero[1], nonzero[2]):
            if heatmap[p][y][x] <= 0:
                continue
            paf[p*2+0][y][x] /= heatmap[p][y][x] #
            paf[p*2+1][y][x] /= heatmap[p][y][x]
        paf = paf.transpose((1, 2, 0))

    def put_vectormap(self, plane_idx, center_from, center_to, heatmap, paf, threshold=8):
        _, height, width = paf.shape[:3]
        vec_x = center_to[0] - center_from[0] #lenth of vector x
        vec_y = center_to[1] - center_from[1] # lenth of vector y

        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

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

        dist = np.abs(bec_x * vec_y - bec_y[:,np.newaxis] * vec_x) 

        paf[plane_idx*2+0][min_y:max_y,min_x:max_x][dist<=threshold] = vec_x
        paf[plane_idx*2+1][min_y:max_y,min_x:max_x][dist<=threshold] = vec_y
        

