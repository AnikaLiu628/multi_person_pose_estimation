
import pickle
import tensorflow as tf
import cv2, os
import numpy as np
import time
import logging
import argparse
import json, re
# from tensorflow.python.client import timeline
from tqdm import tqdm
from common import  CocoPairsRender, read_imgfile, CocoColors, estimate_pose
# from estimator import PoseEstimator , TfPoseEstimator
# from networks import get_network
# from tf_tensorrt_convert import * 
# from pose_dataset import CocoPose
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from estimator import write_coco_json
from network_mobilenet_thin import MobilenetNetworkThin


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True

def round_int(val):
    return int(round(val))

def write_coco_json(human, image_w, image_h):
    '''
    Get the keypoints of a human and convert them to the format required by COCO PythonAPI.

    Args:
        human: A Human object.
        image_w: Width of the image.
        image_h: Height of the image.

    Returns:
        keypoints: Keypoints of the human in the format required by COCO PythonAPI.
    '''
    keypoints = []
    coco_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # print('human_body_parts.keys(): ',human.keys())

    for coco_id in coco_ids:
        if coco_id not in human.keys():
            keypoints.extend([0, 0, 0])
            continue
        body_part = human[coco_id]
        body_part = body_part[1]
        # print(body_part)
        keypoints.extend([round_int(body_part[0] * image_w), round_int(body_part[1] * image_h), 2])
    return keypoints

def compute_oks(keypoints, anns):
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    max_score = 0
    max_visible = []
    for ann in anns:
        score = 0
        gt = ann['keypoints']
        visible = gt[2::3]
        if np.sum(visible) == 0:
            continue
        else:
            gt_point = np.array([(x, y) for x , y in zip(gt[0::3], gt[1::3])])
            pred = np.array([(x, y) for x , y in zip(keypoints[0::3], keypoints[1::3])])
            # import pdb; pdb.set_trace()
            dist = (gt_point - pred) ** 2
            dist = np.sum(dist, axis=1)
            sp = (ann['area'] + np.spacing(1))

        dist[visible == 0] = 0.0
        # dist[visible_pred == 0] = 0.0
        score = np.exp(-dist / vars / 2.0 / sp)

        score = np.mean(score)
        if max_score < score:
            max_score = score
            max_visible = visible
    return max_score, max_visible


def compute_ap(score, threshold =0.5):
    b =  [ 1 if x > threshold else 0 for x in score]
    return np.sum(b)/1.0/len(score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/p2.jpg')
    parser.add_argument('--input-width', type=int, default=432)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--use_tensorrt', type=bool, default=False)
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='mobilenet_v1 / mobilenet_v2 / shufflenet_v2 / mobilenet_thin / mobilenet_thin_s2d1 / mobilenet_thin_FPN / mobilenet_thin_add_more_layers / mobilenet_thin_out4 / hrnet_tiny / higher_hrnet')
    parser.add_argument('--engine',  type=str, default="mobilepose_thin_656x368.engine")
    parser.add_argument('--half16', type=bool, default=False)
    parser.add_argument('--graph', type=str, default="../models/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27/output_model_1301000/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27.pb")
    parser.add_argument('--image_dir', type=str, default='/datasets/t2/data/coco/archive/val2014/')
    parser.add_argument('--coco_json_file', type = str, default = '/datasets/t2/data/coco/archive/annotations/person_keypoints_val2014.json')
    parser.add_argument('--display', type=bool, default=False)
    args = parser.parse_args()

    log_output_path = 'json/COCO_eval_%s_%s_%d_%d.log'%(args.model, args.graph.split('/')[2], args.input_width, args.input_height)
    logging.basicConfig(filename=log_output_path,level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    write_json = 'json/%s_%s_%d_%d.json' %(args.model, args.graph.split('/')[2], args.input_width, args.input_height)
    tf.reset_default_graph()

    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()

    with open('{}'.format(args.graph), 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[2]
    cocoGt = COCO(args.coco_json_file)
    keys = list(cocoGt.imgs.keys())
    catIds = cocoGt.getCatIds(catNms = ['person'])
    imgIds = cocoGt.getImgIds(catIds = catIds)
    keys = list(cocoGt.imgs.keys())

    inputs = tf.get_default_graph().get_tensor_by_name('image:0')
    outputs = tf.get_default_graph().get_tensor_by_name('feat_concat:0')
    heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('hm_out:0')
    pafs_tensor = tf.get_default_graph().get_tensor_by_name('paf_out:0')

    if not os.path.exists(write_json):
        # input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')
        fp = open(write_json, 'w')
        result = []
        
        with tf.Session(config=config) as sess:

            for i, image_id in enumerate(tqdm(keys)):
                #image_id = int(getLastName(img))
                img_meta = cocoGt.imgs[image_id]
                img_idx = img_meta['id']
                ann_idx = cocoGt.getAnnIds(imgIds=image_id)
                anns = cocoGt.loadAnns(ann_idx)

                item = {
                    'image_id':1,
                    'category_id':1,
                    'keypoints':[],
                    'score': 0.0
                }
                img_name = args.image_dir +  'COCO_val2014_%012d.jpg' % image_id
                # print(img_name)
                image = read_imgfile(img_name, args.input_width, args.input_height, web_cam=False)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                
                backbone_feature = sess.run(outputs, feed_dict = {inputs: image})
                heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
                    outputs: backbone_feature
                })

                heatMat, pafMat = heatMat[0], pafMat[0]
                
                humans = estimate_pose(heatMat, pafMat)

                for human in humans :
                    # import pdb; pdb.set_trace();
                    # print(human)
                    res = write_coco_json(human, img_meta['width'], img_meta['height'])
                    item['keypoints'] = res
                    item['image_id'] = image_id
                    item['score'] , visible = compute_oks(res, cocoGt.loadAnns(ann_idx))
                    if len(visible) != 0:
                        for vis in range(17):
                            item['keypoints'][3* vis + 2] = visible[vis]
                        result.append(item)

        json.dump(result,fp)
        fp.close()

    cocoDt = cocoGt.loadRes(write_json)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = keys
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    pred = json.load(open(write_json, 'r'))
    scores = [ x['score'] for x in pred]
    ap50 = compute_ap(scores, 0.5)
    print('ap50 is %f' % ap50)
    logging.info('ap50 is %f' % ap50)
    ap = 0
    for i in np.arange(0.5,1, 0.05).tolist():
        ap = ap + compute_ap(scores, i)
    ap = ap / len(np.arange(0.5, 1 , 0.05).tolist())
    print('ap is %f' % ap)
    logging.info('ap is %f' % ap)
