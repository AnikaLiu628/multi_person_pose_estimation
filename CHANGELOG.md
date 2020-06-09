# Changelog #
## (2020-06-03) ##
### [`MPPE_SRNET_MSE_COCO_368_432_v16`]
(setting/MPPE_SRNET_MSE_COCO_368_432_v16.md)###

[Training loss diagram](logs/MPPE_SRNET_MSE_COCO_368_432_v16.png) | [Log file](logs/MPPE_SRNET_MSE_COCO_368_432_v16.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Model: SRNet

## (2020-06-03) ##
### [`MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v14`]
(setting/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v14.md)###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v14.png) | [Log file](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v14.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and PAFs
* Model: pretrain mobilenetv1(depth=1.0) with pafmodel


## (2020-06-02) ##
### [`MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v11`]
(setting/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v11.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v11.png) | [Log file](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v11.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and 36 PAFs
* Model: Mobilenet_thin

## (2020-06-01) ##
### [`MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v9`]
(setting/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v9.md) ###
[Training loss diagram](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v9.png) | [Log file](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v9.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and 20 PAFs
* Model: pretrain v8, train with gk2th2 data

## (2020-05-22) ##
### [`MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v7`](setting/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v7.md) ###
[Training loss diagram](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v7.png) | [Log file](logs/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v7.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and PAFs
* Model: Mobilenet_thin

## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v11`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v11.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v11.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v11.log)

**Implemented enhancements:**

* **In oerder to reduce training time**
* Heatmap's Guassian kernel size: sigma=3
* MultiGPU train (increase training time)
* Model: Shufflenet_v2

## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v10`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v10.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v10.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v10.log)

**Implemented enhancements:**

* Reduce heatmap's Guassian kernel size: sigma=dynamic
	1. 	Single PE's feature size is (80,64)=5120 pixels, guassian kenerl size is 7 pixels edge 
	2. 	Through PE feature's size as stardard, image's bbox pixels/5120 as scale for gaussain kernel size
* Model: Shufflenet_v2

## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v9`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v9.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v9.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v9.log)

**Implemented enhancements:**

* Reduce heatmap's Guassian kernel size: sigma=6
* Model: Shufflenet_v2

## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v8`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v8.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v8.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v8.log)

**Implemented enhancements:**

* Reduce heatmap's Guassian kernel size: sigma=7
* Model: Shufflenet_v2

## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v7`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v7.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v7.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v7.log)

**Implemented enhancements:**

* Reduce heatmap's Guassian kernel size: sigma=5
* Model: Shufflenet_v2


## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v6`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v6.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v6.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v6.log)

**Implemented enhancements:**

* **Training time cost lot:**
	1. 	increase input size
	2. add depth to space layer
	3. reduce mobilenet_v1 layers
* **Do Guassian kernel experiment**
* Reduce heatmap's Guassian kernel size: sigma=3
* Model: Shufflenet_v2 (faster)

## (2020-04-x) ##
### [`MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v4`](setting/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v4.md) ###
[Training loss diagram](logs/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v4.png) | [Log file](logs/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v4.log)

**Implemented enhancements:**

* **In oerder to increase output size**
* Reduce mobilenet_v1 layers
* Model: Mobilenet_v1

## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v3`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v3.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v3.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v3.log)

**Implemented enhancements:**

* **In oerder to increase output size**
* Add depth to space layer before output layer
* Model: Shufflenet_v2


## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_400_640_v1`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_400_640_v1.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_400_640_v1.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_400_640_v1.log)

**Implemented enhancements:**

* **In oerder to increase output size**
* Increase input size (400, 640)
* Model: Shufflenet_v2


## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v2`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v2.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v2.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v2.log)

**Implemented enhancements:**

* **Through visualize output feature, discover 1. gaussian kernel too big 2. output feautre size too small**
* Reduce heatmap's Guassian kernel size: sigma=1
* Model: Shufflenet_v2

## (2020-04-x) ##
### [`MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v1`](setting/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v1.md) ###
[Training loss diagram](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v1.png) | [Log file](logs/MPPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_v1.log)

**Implemented enhancements:**

* Model: Shufflenet_v2

## (2020-04-x) ##
### [`MPPE_MOBILENET_V2_1.0_MSE_COCO_360_640_v1`](setting/MPPE_MOBILENET_V2_1.0_MSE_COCO_360_640_v1.md) ###
[Training loss diagram](logs/MPPE_MOBILENET_V2_1.0_MSE_COCO_360_640_v1.png) | [Log file](logs/MPPE_MOBILENET_V2_1.0_MSE_COCO_360_640_v1.log)

**Implemented enhancements:**

* Model: Mobilenet_v2

## (2020-04-16) ##
### [`MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1`](setting/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1.md) ###
[Training loss diagram](logs/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1.png) | [Log file](logs/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1.log)

**Implemented enhancements:**

* Add a keypoint between left and right shoulders in COCO dataset
* Input size: (360, 640)
* Preprcoess feature: 
	* heatmap: sigma=8
	* vectormap
* Model: Mobilenet_v1