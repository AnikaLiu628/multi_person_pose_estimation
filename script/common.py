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

CocoPairs = [
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (6, 12), (5, 11), (0, 1), (1, 3), (3, 5), (0, 2), (2, 4), (4, 6)
]
CocoPairsRender = CocoPairs[:-6]

CocoPairsNetwork = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37)
 ]  

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


NMS_Threshold = 0.8 
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4 
Min_Subset_Score = 0.8 
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
    heatmap[heatmap < threshold] = 0 
    part_candidates = heatmap*(heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))
    return part_candidates


def estimate_pose(heatMat, pafMat):
    if heatMat.shape[2] == 17:
        heatMat = np.rollaxis(heatMat, 2, 0)
    if pafMat.shape[2] == 36:
        pafMat = np.rollaxis(pafMat, 2, 0)

    # reliability issue.
    heatMat = heatMat - heatMat.min(axis=1).min(axis=1).reshape(17, 1, 1)
    heatMat = heatMat - heatMat.min(axis=2).reshape(17, heatMat.shape[1], 1)

    _NMS_Threshold = max(np.average(heatMat) * 4.0, NMS_Threshold)
    _NMS_Threshold = min(_NMS_Threshold, 0.3)

    k=0
    coords = [] 
    for heatmap in heatMat[:]: 
        part_candidates = non_max_suppression(heatmap, 5, _NMS_Threshold)
        coords.append(np.where(part_candidates >= _NMS_Threshold))
        k+=1

    connection_all = []
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
        connection = estimate_pose_pair(coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        connection_all.extend(connection)
    
    conns_by_human = dict()
    for idx, c in enumerate(connection_all):
        conns_by_human['human_%d' % idx] = [c]

    # Merging
    no_merge_cache = defaultdict(list)
    empty_set = set()
    while True:
        is_merged = False
        for h1, h2 in itertools.combinations(conns_by_human.keys(), 2):
            if h1 == h2: 
                continue
            if h2 in no_merge_cache[h1]:
                continue
            for c1, c2 in itertools.product(conns_by_human[h1], conns_by_human[h2]): 
                if set(c1['uPartIdx']) & set(c2['uPartIdx']) != empty_set:
                    # if abs(int(c1['uPartIdx'][2:])-int(c2['uPartIdx'][2:])) > 10:
                    #     break
                    # else:
                    is_merged = True
                    conns_by_human[h1].extend(conns_by_human[h2])
                    conns_by_human.pop(h2) 
                    break
            if is_merged:
                no_merge_cache.pop(h1, None)
                break
            else:
                no_merge_cache[h1].append(h2)

        if not is_merged: 
            break
    # reject by subset count
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if len(conns) >= Min_Subset_Cnt}
    # reject by subset max score
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if max([conn['score'] for conn in conns]) >= Min_Subset_Score}
    # list of humans
    humans = [human_conns_to_human_parts(human_conns, heatMat) for human_conns in conns_by_human.values()]
    
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
            elif abs(x1-x2)>=17:
                continue
            connection_temp.append({
                'score': score,
                'coord_p1': (x1, y1),
                'coord_p2': (x2, y2),
                'idx': (idx1, idx2), 
                'partIdx': (partIdx1, partIdx2), 
                'uPartIdx': ('{}-{}-{}'.format(x1, y1, partIdx1), '{}-{}-{}'.format(x2, y2, partIdx2))
            })
    # linear intergral -> 'Assignment' 
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

def find_keypoints(human_list, w, h):
    centers = {}
    mul_centers = {}
    j=0
    for human in human_list:
        part_idxs = human.keys()
        for i in range(17):
            if j == 0:
                if i not in part_idxs:
                    mul_centers[i] = (int(0),int(0))
                    continue
                part_coord = human[i][1]
                center = (int(part_coord[0] * w + 0.5), int(part_coord[1] * h + 0.5)) #108, 92
                centers[i] = center
                mul_centers[i]=center
            
            else:
                if i not in part_idxs:
                    mul_centers[j*17+i] = (int(0),int(0))
                    continue
                part_coord = human[i][1]
                center = (int(part_coord[0] * w + 0.5), int(part_coord[1] * h + 0.5))
                centers[i] = center
                mul_centers[j*17+i]=center
        j+=1
    
    list_points=[]
    visual_point = []
    for c in range(len(mul_centers)):
        list_points.append(mul_centers[c][0])
        list_points.append(mul_centers[c][1])
        visual_point.append((mul_centers[c][1], mul_centers[c][0]))

    return list_points
    # return visual_point
    # return visual_point, list_points


def compute_iou2oks(dts, gts, gt):
    oks = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    variances = (sigmas * 2)**2
    
    if len(dts) * len(gts) == 0:
        return np.array(0), np.array(0)

    oks_mat = np.zeros((len(dts), len(gts)))
    
    #gt
    if(gt==0):
        xg = gts[0::2]; yg = gts[1::2]; vg = []
        for i in range(len(xg)):
            if(xg[i] == 0 and yg[i] ==0):
                vg.append(0)
            else:
                vg.append(1)
    else:
        xg = gts[0::3]; yg = gts[1::3]; vg = gts[2::3]
    xg, yg, vg = np.array(xg), np.array(yg), np.array(vg)

    n_ppl = len(xg)//17
    gt_bboxs = []
    for j in range(n_ppl):
        gt_bboxs.append((xg[j*17:j*17+17].min(),  
                         yg[j*17:j*17+17].min(), 
                         xg[j*17:j*17+17].max(),
                         yg[j*17:j*17+17].max()))
    
    #pred
    xd = dts[0::2]; yd = dts[1::2]
    xd, yd = np.array(xd), np.array(yd)
    pred_n_ppl = len(xd)//17
    dt_bboxs = []
    for j in range(pred_n_ppl):
        dt_bboxs.append((xd[j*17:j*17+17].min(),  
                         yd[j*17:j*17+17].min(), 
                         xd[j*17:j*17+17].max(),
                         yd[j*17:j*17+17].max()))
    
    iou_mat = np.zeros(n_ppl)
    oks_mat = np.zeros(n_ppl)
    for j in range(n_ppl):
        ious = np.zeros(pred_n_ppl)
        for i in range(pred_n_ppl):
            # determine the (x, y)-coordinates of the intersection rectangle
            boxA = gt_bboxs[j]
            boxB = dt_bboxs[i]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            ious[i] = interArea / float(boxAArea + boxBArea - interArea)
        iou_mat[j] = ious.max()
        idx = np.argmax(ious) #dt index -> gt
        #gt
        gt_x = xg[j*17:j*17+17]; gt_y = yg[j*17:j*17+17]; gt_v = vg[j*17:j*17+17]
        k1 = np.count_nonzero(gt_v > 0)
        x0 = gt_bboxs[j][0]; x1 = gt_bboxs[j][2]
        y0 = gt_bboxs[j][1]; y1 = gt_bboxs[j][3]
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        #dt
        dt_x = xd[idx*17:idx*17+17]; dt_y = yd[idx*17:idx*17+17]
        
        oksup = np.zeros(17)
        dx = abs(dt_x - gt_x)
        dy = abs(dt_y - gt_y)
        e = (dx**2 + dy**2) / variances / (area + np.spacing(1)) / 2
        if (np.sum(np.exp(-e)) / k1) <= 0.3:
            oks_mat[j] = 0.0    
        else:
            oks_mat[j] = np.sum(np.exp(-e)) / k1 
        
    return oks_mat, iou_mat