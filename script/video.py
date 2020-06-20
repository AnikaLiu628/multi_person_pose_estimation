import tensorflow as tf
import cv2
import numpy as np
import argparse

from common import estimate_pose, draw_humans, read_imgfile

import time


fps_time = 0

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
parser.add_argument('--input_width', type=int, default=432)
parser.add_argument('--input_height', type=int, default=368)
parser.add_argument('--model', type=str, default='MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19')
parser.add_argument('--checkpoint', type=str, default='522000')
parser.add_argument('--video', type=str, default='../data/video/A043_P003_C005_D002_S012.avi')
args = parser.parse_args()

video_name = args.video.split('/')[3].split('.')[0]
t0 = time.time()

tf.reset_default_graph()

from tensorflow.core.framework import graph_pb2
graph_def = graph_pb2.GraphDef()
# Download model from https://www.dropbox.com/s/2dw1oz9l9hi9avg/optimized_openpose.pb
with open('../models/{}/output_model_{}/{}.pb'.format(args.model, args.checkpoint, args.model), 'rb') as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')

VideoWriter = cv2.VideoWriter('../data/video_record/output_{}_{}.avi'.format(video_name, args.model), cv2.VideoWriter_fourcc(*'XVID'), 1, (853, 480))
t1 = time.time()
print(t1 - t0)

inputs = tf.get_default_graph().get_tensor_by_name('image:0')
# outputs = tf.get_default_graph().get_tensor_by_name('shufflenet_v2/conv5/Relu:0')
# outputs = tf.get_default_graph().get_tensor_by_name('PixelShuffle1/depth_to_space:0')
outputs = tf.get_default_graph().get_tensor_by_name('feat_concat:0')
# heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('paf/class_out:0')
heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('hm_out:0')
# pafs_tensor = tf.get_default_graph().get_tensor_by_name('paf/regression_out:0')
pafs_tensor = tf.get_default_graph().get_tensor_by_name('paf_out:0')

cap = cv2.VideoCapture(args.video)
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        t2 = time.time()
        print('rea model time cost:   ', t2 - t1)

        image = read_imgfile(img, args.input_width, args.input_height, web_cam=True)
        # img_np = image.copy()

        t3 = time.time()
        print('read image time cost:    ', t3 - t2)

        with tf.Session() as sess:
            backbone_feature = sess.run(outputs, feed_dict = {inputs: image})
            heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
                outputs: backbone_feature
            })

            t4 = time.time()
            print('feature out time cost:   ', t4 - t3)

            heatMat, pafMat = heatMat[0], pafMat[0]

            humans = estimate_pose(heatMat, pafMat)

            # display
            # image = cv2.imread(args.imgpath)
            image = img
            image_h, image_w = image.shape[:2]
            image = draw_humans(image, humans)

            scale = 480.0 / image_h
            newh, neww = 480, int(scale * image_w + 0.5)

            image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

            # cv2.imshow('result', image)
            # t5 = time.time()
            # print('estimate human pose: ', t5 - t4)
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            fps_time = time.time()
            cv2.imshow('template', image)
            VideoWriter.write(image)
            if cv2.waitKey(1) == 27:
                break
