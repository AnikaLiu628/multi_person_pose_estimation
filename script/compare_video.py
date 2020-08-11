import tensorflow as tf
import os
import cv2
import numpy as np
import argparse

from common import estimate_pose, draw_humans, read_imgfile

import time


fps_time = 0

parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
parser.add_argument('--input_width', type=int, default=432)
parser.add_argument('--input_height', type=int, default=368)
parser.add_argument('--model1', type=str, default='MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19')
parser.add_argument('--model2', type=str, default='MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19')
parser.add_argument('--checkpoint1', type=str, default='522000')
parser.add_argument('--checkpoint2', type=str, default='522000')
parser.add_argument('--video', type=str, default='../data/video/A043_P003_C005_D002_S012.avi')
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

MP_POSE1 = '../models/{}/output_model_{}/{}.pb'.format(args.model1, args.checkpoint1, args.model1)
MP_POSE2 = '../models/{}/output_model_{}/{}.pb'.format(args.model2, args.checkpoint2, args.model2)

detection_graph1 = tf.Graph()
detection_graph2 = tf.Graph()


with detection_graph1.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MP_POSE1, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph2.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MP_POSE2, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess1 = tf.Session(graph=detection_graph1, config=config)
sess2 = tf.Session(graph=detection_graph2, config=config)

inputs1 = detection_graph1.get_tensor_by_name('image:0')
outputs1 = detection_graph1.get_tensor_by_name('feat_concat:0')
heatmaps_tensor1 = detection_graph1.get_tensor_by_name('hm_out:0')
pafs_tensor1 = detection_graph1.get_tensor_by_name('paf_out:0')

inputs2 = detection_graph2.get_tensor_by_name('image:0')
outputs2 = detection_graph2.get_tensor_by_name('feat_concat:0')
heatmaps_tensor2 = detection_graph2.get_tensor_by_name('hm_out:0')
pafs_tensor2 = detection_graph2.get_tensor_by_name('paf_out:0')

video_path = args.video #'/DataCenter/t3/data/ActionRecongnition/flow/archive/RGBvideo/'
VideoList = os.listdir(video_path)

for v in VideoList:
    vid_name = os.path.join(video_path,v)
    video_name = vid_name.split('/')[-1].split('.')[0]
    # file1 = open("label_video.txt","a")
    # VideoWriter = cv2.VideoWriter('../data/video_record/output_{}_{}.avi'.format(video_name, args.model), cv2.VideoWriter_fourcc(*'XVID'), 1, (853, 480))
    cap = cv2.VideoCapture(vid_name)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:

            image = read_imgfile(img, args.input_width, args.input_height, web_cam=True)
        
            backbone_feature1 = sess1.run(outputs1, feed_dict = {inputs1: image})
            heatMat1, pafMat1 = sess1.run([heatmaps_tensor1, pafs_tensor1], feed_dict={
                outputs1: backbone_feature1
            })

            heatMat1, pafMat1 = heatMat1[0], pafMat1[0]

            humans1 = estimate_pose(heatMat1, pafMat1)

            backbone_feature2 = sess2.run(outputs2, feed_dict = {inputs2: image})
            heatMat2, pafMat2 = sess2.run([heatmaps_tensor2, pafs_tensor2], feed_dict={
                outputs2: backbone_feature2
            })

            heatMat2, pafMat2 = heatMat2[0], pafMat2[0]

            humans2 = estimate_pose(heatMat2, pafMat2)

            # display
            image = img
            image_h, image_w = image.shape[:2]
            image1 = draw_humans(image, humans1)
            image2 = draw_humans(image, humans2)

            scale = 480.0 / image_h
            newh, neww = 480, int(scale * image_w + 0.5)

            image1 = cv2.resize(image1, (neww, newh), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (neww, newh), interpolation=cv2.INTER_AREA)

            cv2.putText(image1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            fps_time = time.time()
            image_hor = cv2.hconcat([image1, image2])
            cv2.imshow('{}-{} / {}-{}'.format(args.model1, args.checkpoint1, args.model2, args.checkpoint2), image_hor)
            # VideoWriter.write(image)
            cv2.waitKey(1)
        else:
            print('video STOP...{}'.format(video_name))
            
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     print('skip {}'.format(video_name))
            #     cap.release()
            #     # VideoWriter.release()
            #     cv2.destroyAllWindows()
            # else:
            #     print('write {}'.format(video_name))
            #     # file1.write(video_name+'\n')
            #     # file1.close() 
            #     cap.release()
            #     # VideoWriter.release()
            #     cv2.destroyAllWindows()

            cap.release()
            cv2.destroyAllWindows()