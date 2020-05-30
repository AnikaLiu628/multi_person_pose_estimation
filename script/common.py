'''
All code is highly based on Ildoo Kim's code (https://github.com/ildoonet/tf-openpose)
and derived from the OpenPose Library (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
'''

from collections import defaultdict
from enum import Enum
import math

import numpy as np
import itertools
import cv2
from scipy.ndimage.filters import maximum_filter


class CocoPart(Enum):
    # Nose = 0
    # Neck = 1
    # RShoulder = 2
    # RElbow = 3
    # RWrist = 4
    # LShoulder = 5
    # LElbow = 6
    # LWrist = 7
    # RHip = 8
    # RKnee = 9
    # RAnkle = 10
    # LHip = 11
    # LKnee = 12
    # LAnkle = 13
    # REye = 14
    # LEye = 15
    # REar = 16
    # LEar = 17
    # Background = 18
    Nose = 0
    REye = 1
    LEye = 2
    REar = 3
    LEar = 4
    RShoulder = 5
    LShoulder = 6
    RElbow = 7
    LElbow = 8
    RWrist = 9
    LWrist = 10
    RHip = 11
    LHip = 12
    RKnee = 13
    LKnee = 14
    RAnkle = 15
    LAnkle = 16

# CocoPairs = [
#     (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
#     (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
# ]   # = 19 [[5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16], [5, 6], [11, 12]]
CocoPairs = [
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12)
]
CocoPairsRender = CocoPairs[:]
# CocoPairsNetwork = [
#     (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
#     (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
#  ]  
CocoPairsNetwork = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)
 ]  

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


NMS_Threshold = 0.8
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4
Min_Subset_Score = 0.8 #0.8
Max_Human = 96


def human_conns_to_human_parts(human_conns, heatMat):
    human_parts = defaultdict(lambda: None)
    for conn in human_conns:
        human_parts[conn['partIdx'][0]] = (
            conn['partIdx'][0], # part index kps, only arms
            (conn['coord_p1'][0] / heatMat.shape[2], conn['coord_p1'][1] / heatMat.shape[1]), # relative coordinates
            heatMat[conn['partIdx'][0], conn['coord_p1'][1], conn['coord_p1'][0]] # score
            )
        human_parts[conn['partIdx'][1]] = (
            conn['partIdx'][1],
            (conn['coord_p2'][0] / heatMat.shape[2], conn['coord_p2'][1] / heatMat.shape[1]),
            heatMat[conn['partIdx'][1], conn['coord_p2'][1], conn['coord_p2'][0]]
            )
    return human_parts


def non_max_suppression(heatmap, window_size=3, threshold=NMS_Threshold):
    heatmap[heatmap < threshold] = 0 # set low values to 0
    part_candidates = heatmap*(heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))
    # print(np.where(part_candidates >= threshold))
    return part_candidates


def estimate_pose(heatMat, pafMat):
    if heatMat.shape[2] == 17:
        # transform from [height, width, n_parts] to [n_parts, height, width]
        heatMat = np.rollaxis(heatMat, 2, 0)
    if pafMat.shape[2] == 20:
        # transform from [height, width, 2*n_pairs] to [2*n_pairs, height, width]
        pafMat = np.rollaxis(pafMat, 2, 0)

    # reliability issue.
    heatMat = heatMat - heatMat.min(axis=1).min(axis=1).reshape(17, 1, 1)
    heatMat = heatMat - heatMat.min(axis=2).reshape(17, heatMat.shape[1], 1)

    _NMS_Threshold = max(np.average(heatMat) * 4.0, NMS_Threshold)
    _NMS_Threshold = min(_NMS_Threshold, 0.3)

    #挑出hm中可能是kps的點 [17, (y, x)] <-可以多個
    k=0
    coords = [] # for each part index, it stores coordinates of candidates
    for heatmap in heatMat[:]: # heatMat = (17, 46, 54) # find the max. location by NMS
        part_candidates = non_max_suppression(heatmap, 5, _NMS_Threshold)
        coords.append(np.where(part_candidates >= _NMS_Threshold))
        k+=1

    #計算linear intergral, assignment
    connection_all = [] # all connections detected. no information about what humans they belong to
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
        connection = estimate_pose_pair(coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        connection_all.extend(connection)
    # example: connection_all = [{'score': 9.898817479610443, 'coord_p1': (22, 20), 'coord_p2': (23, 24), 'idx': (2, 2), 'partIdx': (5, 7), 'uPartIdx': ('22-20-5', '23-24-7')}, {.}, {.}...]

    #給idx
    conns_by_human = dict()
    for idx, c in enumerate(connection_all):
        conns_by_human['human_%d' % idx] = [c] # at first, all connections belong to different humans
    # example: conns_by_human = {'human_0': [{'score': 9.898817479610443, 'coord_p1': (22, 20), 'coord_p2': (23, 24), 
    #                                           'idx': (2, 2), 'partIdx': (5, 7), 'uPartIdx': ('22-20-5', '23-24-7')}], 
    #                            'human_1'[{...}], ...} #gives idx human%d

    # Merging
    no_merge_cache = defaultdict(list)
    empty_set = set()
    while True:
        is_merged = False
        for h1, h2 in itertools.combinations(conns_by_human.keys(), 2): #所有可能的連線 h1->所有h2, h2->所有h1  #itertools.combinations(range(3), 2) 求列表或生成器中指定數目的元素不重複的所有組合
            # conns_by_human.keys() = ['human1', 'human2', ...]
            if h1 == h2: 
                continue
            # print('no_merge_cache[h1]:  ', no_merge_cache[h1])
            if h2 in no_merge_cache[h1]:
                continue
            for c1, c2 in itertools.product(conns_by_human[h1], conns_by_human[h2]): #產生多個列表和迭代器的(積)
                # print('conns_by_human[h1]:  ', conns_by_human[h1])
                # print('conns_by_human[h2]:  ', conns_by_human[h2])
                # if two humans share a part (same part idx and coordinates), merge those humans
                if set(c1['uPartIdx']) & set(c2['uPartIdx']) != empty_set:
                    # print(c1['uPartIdx'])
                    # if abs(int(c1['uPartIdx'][2:])-int(c2['uPartIdx'][2:])) > 10:
                    #     break
                    # else:
                    is_merged = True
                    # extend human1 connectios with human2 connections
                    conns_by_human[h1].extend(conns_by_human[h2])
                    conns_by_human.pop(h2) # delete human2
                    break
                # print('conns_by_human:  ', conns_by_human)
            if is_merged:
                no_merge_cache.pop(h1, None)
                break
            else:
                no_merge_cache[h1].append(h2)

        if not is_merged: # if no more mergings are possible, then break
            break
    # print('conns_by_human.items():  ', conns_by_human.items())
    # reject by subset count
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if len(conns) >= Min_Subset_Cnt}
    # reject by subset max score
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if max([conn['score'] for conn in conns]) >= Min_Subset_Score}

    # print('conns_by_human:  ', conns_by_human)
    # list of humans
    humans = [human_conns_to_human_parts(human_conns, heatMat) for human_conns in conns_by_human.values()]
    # print(humans)
    return humans


def estimate_pose_pair(coords, partIdx1, partIdx2, pafMatX, pafMatY):
    connection_temp = [] # all possible connections
    peak_coord1, peak_coord2 = coords[partIdx1], coords[partIdx2]

    for idx1, (y1, x1) in enumerate(zip(peak_coord1[0], peak_coord1[1])):
        for idx2, (y2, x2) in enumerate(zip(peak_coord2[0], peak_coord2[1])):
            score, count = get_score(x1, y1, x2, y2, pafMatX, pafMatY)
            if (partIdx1, partIdx2) in [(6, 8), (8, 10), (5, 7), (7, 9)]: # arms
                if count < InterMinAbove_Threshold // 2 or score <= 0.0:
                    continue
            elif count < InterMinAbove_Threshold or score <= 0.0:
                continue
            elif abs(x1-x2)>=12:
                continue
            # print((x1, y1), (x2, y2), (partIdx1, partIdx2))
            print('score:   ', score)
            print('coord_p1:    ', (x1, y1))
            print('coord_p2:    ', (x2, y2))
            print('idx: ', (idx1, idx2))
            print('partIdx: ', (partIdx1, partIdx2))
            connection_temp.append({
                'score': score,
                'coord_p1': (x1, y1),
                'coord_p2': (x2, y2),
                'idx': (idx1, idx2), # connection candidate identifier, NMS找出的許多點 哪個（idx）做連結
                'partIdx': (partIdx1, partIdx2), 
                'uPartIdx': ('{}-{}-{}'.format(x1, y1, partIdx1), '{}-{}-{}'.format(x2, y2, partIdx2))
            })
    # linear intergral -> 'Assignment' 把score由大排到小 選出最大score的連結 可以塞掉一些連結的candidate 
    connection = []
    used_idx1, used_idx2 = [], []
    # sort possible connections by score, from maximum to minimum
    for conn_candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
        # check not connected
        if conn_candidate['idx'][0] in used_idx1 or conn_candidate['idx'][1] in used_idx2:
            continue
        connection.append(conn_candidate)
        used_idx1.append(conn_candidate['idx'][0])
        used_idx2.append(conn_candidate['idx'][1])
    return connection

#'linear intergral'
def get_score(x1, y1, x2, y2, pafMatX, pafMatY):
    #sampling
    num_inter = 10
    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2)

    if normVec < 1e-4:
        return 0.0, 0

    vx, vy = dx / normVec, dy / normVec

    xs = np.arange(x1, x2, dx / num_inter) if x1 != x2 else np.full((num_inter, ), x1) #如果 x1 -> x2 是相同的 除去
    ys = np.arange(y1, y2, dy / num_inter) if y1 != y2 else np.full((num_inter, ), y1)
    xs = (xs + 0.5).astype(np.int8)
    ys = (ys + 0.5).astype(np.int8)

    # without vectorization
    # pafXs = np.zeros(num_inter)
    # pafYs = np.zeros(num_inter)
    # for idx, (mx, my) in enumerate(zip(xs, ys)):
    #     pafXs[idx] = pafMatX[my][mx]
    #     pafYs[idx] = pafMatY[my][mx]

    # vectorization slow?
    pafXs = pafMatX[ys, xs]
    pafYs = pafMatY[ys, xs]

    local_scores = pafXs * vx + pafYs * vy
    thidxs = local_scores > Inter_Threashold

    return sum(local_scores * thidxs), sum(thidxs)


def read_imgfile(path, width, height, web_cam = False):
    if web_cam is True:
        img = path
    else:
        img = cv2.imread(path)
    val_img = preprocess(img, width, height)
    return val_img


def preprocess(img, width, height):
    val_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads in BGR format
    val_img = cv2.resize(val_img, (width, height)) # each net accept only a certain size
    val_img = val_img.reshape([1, height, width, 3])
    val_img = val_img.astype(float)
    # val_img = val_img * (2.0 / 255.0) - 1.0 # image range from -1 to +1
    val_img= (val_img - 127.5) * 0.0078125
    return val_img


def draw_humans(img, human_list):
    img_copied = np.copy(img)
    image_h, image_w = img_copied.shape[:2]
    centers = {}
    for human in human_list:
        part_idxs = human.keys()

        # draw point
        for i in range(17):
            if i not in part_idxs:
                continue
            part_coord = human[i][1]
            center = (int(part_coord[0] * image_w + 0.5), int(part_coord[1] * image_h + 0.5))
            centers[i] = center
            cv2.circle(img_copied, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in part_idxs or pair[1] not in part_idxs:
                continue

            img_copied = cv2.line(img_copied, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return img_copied
