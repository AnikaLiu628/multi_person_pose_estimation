Multi-person Pose Estimation
=========

Dataset
-----------------------
* Since there is a problem with heavy data while preprocessor loading in multipose estimation experiments,the  data process part is highly recommended to do respectively. Hence, there is `Coco (rotated 90, 180, 270, flip) train` TFrecord.

Fuse `Coco`, `Panoptic`, `MPII`, `PoseTrack` datasets for training dataset.

|Dataset|Image|
| ----|----|
|Coco train|56,600 / 118,287|
|Coco rotated 90 train|56,600 / 118,287|
|Coco rotated 180 train|56,600 / 118,287|
|Coco rotated 270 train|56,600 / 118,287|
|Coco flip train|56,600 / 118,287|
|Coco val|2,346 / 5,000|
|MPII train(human_pose_v1_u12_2)|- / 24,984|
|Panoptic(171026pose3)|--|
|Panoptic filtered 5 points(171026pose3)|222,611 / ------|
|Panoptic train|17,565,778 / 18,089,009|
|Panoptic train(w/o val)|11,541,269 / 11,948,561|
|Panoptic val|5,556 / 5,556|
|Panoptic test|2,066 / 2,066|

|Fused datasets|Scale|Guassian kernel|Content|
| ----|----|----|----|
|F6|4|**kernel_sigma**: bbox area/image area[>0.32=3, 0.32~0.04=3, <0.04=2]|Coco train + Coco rotated 90 train + Coco rotated 180 train + Coco rotated 270 train + PoseTrack train|
|F7|4|**kernel_sigma**: bbox area/image area[>0.32=3, 0.32~0.04=3, <0.04=2]|Coco train + Coco rotated 90 train + Coco rotated 180 train + Coco rotated 270 train + PoseTrack train + Ntu_rgb+d|

Train, Test Model
-----------------------

**Backbones of model:**

* MobilenetThin
* Mobilenet v1, v2, v3


|Model|Train|Eval|log|
| ----|----|----|----|
|`MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1`|[train](setting/MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.md)|[eval](script/evaluate.py)|[log](logs/MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.log)|
|`MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1`|[train](setting/MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.md)|[eval](script/evaluate.py)|[log](logs/MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1.log)|
|`MPPE_F6_small_FPN_MOBILENET_THIN_MSE_368_432_v6`|[train](setting/MPPE_F6_small_FPN_MOBILENET_THIN_MSE_368_432_v6.md)|[eval](script/evaluate.py)|[log](logs/MPPE_F6_small_FPN_MOBILENET_THIN_MSE_368_432_v6.log)|  
|`MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1`|[train](setting/MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1.md)|[eval](script/evaluate.py)|[log](logs/MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1.log)|
|`MPPE_F6_MOBILENET_v2_0.5_MSE_368_432_v1`|[train](setting/MPPE_F6_MOBILENET_v2_0.5_MSE_368_432_v1.md)|[eval](script/evaluate.py)|[log](logs/MPPE_F6_MOBILENET_v2_0.5_MSE_368_432_v1.log)|
|`MPPE_F6_SHUFFLENET_v2_0.5_MSE_368_432_v1`|[train](setting/MPPE_F6_SHUFFLENET_v2_0.5_MSE_368_432_v1.md)|[eval](script/evaluate.py)|[log](logs/MPPE_F6_SHUFFLENET_v2_0.5_MSE_368_432_v1.log)|

*Reference: [tfopenpose](https://github.com/ildoonet/tf-pose-estimation/blob/master/etcs/experiments.md)*

Test OKS score on coco validation set
-----------------------
|Model/Error|OKS|num of Human Detect|
| ----|----|----|
|`MPPE_F6_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1`|0.77615|0.41209|
|`MPPE_F7_FPN_MOBILENET_THIN_0.75_MSE_368_432_v1`|0.65504|0.40462|
|`MPPE_F6_small_FPN_MOBILENET_THIN_MSE_368_432_v6`|0.49746|0.28825|
|`MPPE_F6_MOBILENET_v1_0.5_MSE_368_432_v1`|0.57788|0.30845|   
|`MPPE_F6_MOBILENET_v2_0.5_MSE_368_432_v1`|0.47801|0.31142|
|`MPPE_F6_SHUFFLENET_v2_0.5_MSE_368_432_v1`|0.33706|0.25668|