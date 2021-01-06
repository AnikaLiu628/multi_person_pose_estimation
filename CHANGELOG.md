# Changelog #

## (2020-12-24) ##
### [`MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1`](setting/MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1.md) ###

[Training loss diagram](logs/MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1.png) | [Log file](logs/MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Scale: 4
* Data: Coco_orginal + Coco_rotate90 + Coco_rotate270 + Coco_rotate180 + PoseTrack (**kernel_sigma**: bbox area/image area[>0.32=3, 0.32~0.04=3, <0.04=2])
* Model: mobilenetv1

## (2020-12-08) ##
### [`MPPE_F6_small_FPN_MOBILENET_THIN_0.5_MSE_368_432_v1`](setting/MPPE_F6_small_FPN_MOBILENET_THIN_0.5_MSE_368_432_v1.md) ###

[Training loss diagram](logs/MPPE_F6_small_FPN_MOBILENET_THIN_0.5_MSE_368_432_v1.png) | [Log file](logs/MPPE_F6_small_FPN_MOBILENET_THIN_0.5_MSE_368_432_v1.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Scale: 4
* Data: Coco_orginal + Coco_rotate90 + Coco_rotate270 + Coco_rotate180 + PoseTrack (**kernel_sigma**: bbox area/image area[>0.32=3, 0.32~0.04=3, <0.04=2])
* Model: reduce feature number in backbone + FPN


## (2020-12-02) ##
### [`MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1`](setting/MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.md) ###

[Training loss diagram](logs/MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.png) | [Log file](logs/MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Scale: 4
* Data: Coco_orginal + Coco_rotate90 + Coco_rotate270 + Coco_rotate180 + PoseTrack + Ntu_rgb+d
* Model: `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt` + FPN
* Result: In **fall down action**, NTU dataset increase false positive

## (2020-09-25) ##
### [`MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1`](setting/MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.md) ###

[Training loss diagram](logs/MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.png) | [Log file](logs/MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.log)

**Implemented enhancements:**

* Pre-build TFrecord for data preprocessing
	* Scale: 4
* Data: Coco_orginal + Coco_rotate90 + Coco_rotate270 + Coco_rotate180 + PoseTrack (**kernel_sigma**: bbox area/image area[>0.32=3, 0.32~0.04=3, <0.04=2])
* Model: `pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224_2017_06_14.ckpt` + FPN
