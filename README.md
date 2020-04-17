Multi Pose Estimation
=========

Dataset
-----------------------

Fuse `Coco`, `Panoptic`, `PoseTrack`, `Mpii` datasets for training dataset.

|Dataset|Image|
| ----|----|
|Coco train|149,813 / 149,813|
|Coco val|6,352 / 6,352|
|Coco filterd 5 points|126,990 / 149,813|
|Panoptic(171026pose3)|--|
|Panoptic filtered 5 points(171026pose3)|222,611 / ------|
|Panoptic train|17,565,778 / 18,089,009|
|Panoptic train(w/o val)|11,541,269 / 11,948,561|
|Panoptic val|5,556 / 5,556|
|PoseTrack train|73,625 / 99,128|
|PoseTrack val|32,005 / 45,562|
|Mpii train|--/--|
|--|--|

|Fused datasets|Content|
| ----|----|
|F1|Coco train + PoseTrack train|
|F2|Coco train + Panoptic train(w/o val)|
|F3|Coco train + Mpii train|
|F4|Coco train + PoseTrack train + Panoptic train + Mpii train|

Train, Test Model
-----------------------

|Model|Train|Eval|log|
| ----|----|----|----|
|`MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1`|[train](setting/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1.md)|[eval](script/evaluate.py)|[log](logs/MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1.log)|

Test error rate on panoptic validation set(5556)
-----------------------
|Model/Error|Average Error|
| ----|----|


Test error rate on coco validation set(6352)
-----------------------
|Model/Error|Average Error|
| ----|----|
|`MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v1`|-|

