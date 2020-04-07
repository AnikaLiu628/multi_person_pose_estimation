from scipy import ndimage
import math
import numpy as np
import tensorflow as tf

class Postprocess():
    def __init__(self, ):
        pass
    #Non Maximum Suppression
    def NMS(self, heatmap_x):
        window_size = 5
        heatmap = np.zeros_like(heatmap_x[0][0])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            heatmap_x = sess.run(heatmap_x)
            for plane_idx in range(len(heatmap_x)):
                for part_idx in range(18):
                    if part_idx == 0:
                        per_hm = heatmap_x[plane_idx][part_idx]
                        heatmap_max = ndimage.maximum_filter(per_hm, footprint=np.ones((window_size, window_size)))
                        heatmap = np.expand_dims(heatmap_max, axis=0)
                    else:
                        per_hm = heatmap_x[plane_idx][part_idx]
                        heatmap_max = ndimage.maximum_filter(per_hm, footprint=np.ones((window_size, window_size)))
                        heatmap_max = np.expand_dims(heatmap_max, axis=0)
                        heatmap = np.concatenate((heatmap, heatmap_max),axis=0)
                # print('heatmap.shape:   ', heatmap.shape) #18, 46, 80
                if plane_idx == 0:
                    heatmap_max_batch = heatmap
                    heatmap_max_batch = np.expand_dims(heatmap_max_batch, axis=0)
                else:
                    heatmap = np.expand_dims(heatmap, axis=0)
                    heatmap_max_batch = np.concatenate((heatmap_max_batch, heatmap),axis=0)
            # print(heatmap_max_batch.shape) #16, 18, 46, 80
        heatmap_max_batch = tf.convert_to_tensor(heatmap_max_batch)
        return heatmap_max_batch
    
    def np_NMS(self, heatmap_x):
        window_size = 1
        heatmap_x[heatmap_x < 0.1] = 0
        heatmap = np.zeros_like(heatmap_x[0][0])
        for plane_idx in range(len(heatmap_x)):
            for part_idx in range(18):
                if part_idx == 0:
                    per_hm = heatmap_x[plane_idx][part_idx]
                    heatmap_max = ndimage.maximum_filter(per_hm, footprint=np.ones((window_size, window_size)))
                    heatmap = np.expand_dims(heatmap_max, axis=0)
                else:
                    per_hm = heatmap_x[plane_idx][part_idx]
                    heatmap_max = ndimage.maximum_filter(per_hm, footprint=np.ones((window_size, window_size)))
                    heatmap_max = np.expand_dims(heatmap_max, axis=0)
                    heatmap = np.concatenate((heatmap, heatmap_max),axis=0)
            # print('heatmap.shape:   ', heatmap.shape) #18, 46, 80
            if plane_idx == 0:
                heatmap_max_batch = heatmap
                heatmap_max_batch = np.expand_dims(heatmap_max_batch, axis=0)
            else:
                heatmap = np.expand_dims(heatmap, axis=0)
                heatmap_max_batch = np.concatenate((heatmap_max_batch, heatmap),axis=0)
        # print(heatmap_max_batch.shape) #16, 18, 46, 80
        # heatmap_max_batch = tf.convert_to_tensor(heatmap_max_batch)
        return heatmap_max_batch
    #Bipartite graph
    #Assignment
    #Merging
    def BG(self, vectmap_x):
        _, 
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            vectmap_x = sess.run(vectmap_x)
            


'''
#對hm找最大值（Non Maximum Suppression）
#一個kernel掃過的地方 提取最大值
part_candidates = heatmap*(heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))

#把vec連線 Bipartite graph
dx, dy = x2 - x1, y2 - y1
normVec = math.sqrt(dx **2 + dy **2)
vx, vy = dx / normVec, dy / normVec

#sampling
num_samples = 10
xs = np.arange(x1, x2, dx/num_samples).astype(np.int8)
ys = np.arange(y1, y2, dy/num_samples).astype(np.int8)

#eval on the fields
pafXs = pafX[ys, xs]
pafYs = pafY[ys, xs]

#integral
score = sum(pafXs * vx + pafYs *vy) / num_samples

#assignment
connection = []
used_idx1, used_idx2 = [], []
# sort possible connections by score, form max to min
for conn_candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
    #check out connected
    if conn_candidate['idx'][0] in used_idx1 or conn_candidate['idx'][1] in used_idx2:
        continue
    connection.append(conn_candidate)
    used_idx1.append(conn_candidate['idx'][0])
    used_idx2.append(conn_candidate['idx'][1])

#merging 把part連起來
from collections import defaultdict
import itertools

no_merge_cache = defaultdict(list)
empty_set = set()

while True:
    is_merged = False
    for h1, h2 in itertools.combinations(connections_by_human.key(), 2):
        for c1, c2 in itertools.product(connections_by_human[h1], connections_by_human[h2]):
            # if two humans share a part (same part idx and coordinates), merge those humans
            if set(c1['partCoordsAndIdx']) & set(c2['partCoordAndIdx']) != empty_set:
                is_merged = True
                # extend human1 connections with human2 connections
                connections_by_human[h1].extend(connections_by_human[h2])
                connections_by_human.pop(h2) #delete human2
                break
    if not is_merged: #if no more merge possible, then break
        break

#describe humans as a set of pars, not as a set of connections
humans = [human_conn_to_human_parts(human_conns) for human_conns in connections_by_human.value()]
'''