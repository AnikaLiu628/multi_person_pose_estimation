# Changelog #
## (2020-07-09) ##
### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + rotate270 + panoptic + rotate180
* Model: HRnet
* -------fail--------

### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + rotate270 + panoptic + rotate180
* Model: `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt` + FPN, comapre with v27, v26


## (2020-07-01) ##
### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: reduce guassian kernel size = 2 (orginal + rotate90 + rotate270 + rotate180)
* Model: `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt`, comapre with v27 


### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + rotate270 + panoptic + rotate180
* Model: pretrain `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt` compare with v26

### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + rotate270 + panoptic + rotate180
* Model: pretrain `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt` + one space_to_depth layer, compare with v23

## (2020-06-23) ##
### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + rotate270 + panoptic
* Model: pretrain `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt` + one space_to_depth layer, compare with v22

## (2020-06-22) ##
### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + rotate270 + panoptic
* Model: pretrain `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt`

## (2020-06-17) ##
### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + panoptic
* Model: pretrain `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt`

### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v20`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v20.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v20.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v20.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90 + flip
* Model: pretrain `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt`

## (2020-06-17) ##
### [`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19`](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19.md) ###

[Training loss diagram](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19.png) | [Log file](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Input size: (368, 432)
	* Build heatmaps and fullPAFs
* Data: orginal + rotate90
* Model: pretrain `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt`

