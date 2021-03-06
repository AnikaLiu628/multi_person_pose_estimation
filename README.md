Multi Pose Estimation
=========

Dataset
-----------------------
* Since there is a problem with heavy data while preprocessor loading in multipose estimation experiments,the  data process part is highly recommended to do respectively. Hence, there is `Coco (rotated 90, 180, 270, flip) train` TFrecord.

Fuse `Coco`, `Panoptic`, datasets for training dataset.

|Dataset|Image|
| ----|----|
|Coco train|118,287 / 118,287|
|Coco rotated 90 train|118,287 / 118,287|
|Coco rotated 180 train|118,287 / 118,287|
|Coco rotated 270 train|118,287 / 118,287|
|Coco flip train|118,287 / 118,287|
|Coco val|5,000 / 5,000|
|Panoptic(171026pose3)|--|
|Panoptic filtered 5 points(171026pose3)|222,611 / ------|
|Panoptic train|17,565,778 / 18,089,009|
|Panoptic train(w/o val)|11,541,269 / 11,948,561|
|Panoptic val|5,556 / 5,556|
|Panoptic test|2,066 / 2,066|

|Fused datasets|Content|
| ----|----|
|F1|Coco train + Coco rotated 90 train + Coco rotated 180 train + Coco rotated 270 train + Coco flip train(output: 92, 108)|
|-|Coco train + Coco rotated 90 train + Coco rotated 180 train + Coco rotated 270 train + Coco flip train|
|-|Coco train + Coco rotated 90 train + Coco rotated 180 train + Coco rotated 270 train + Panoptic test|

Train, Test Model
-----------------------

|Model|Train|Eval|log|
| ----|----|----|----|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29.log)|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30`|[train](setting/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30.md)|[eval](script/eval.py)|[log](logs/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30.log)|
|`MPPE_F1_MOBILENET_THIN_0.75_MSE_COCO_368_432_v1`|[train](setting/MPPE_F1_MOBILENET_THIN_0.75_MSE_COCO_368_432_v1.md)|[eval](script/eval.py)|[log](logs/MPPE_F1_MOBILENET_THIN_0.75_MSE_COCO_368_432_v1.log)|

Test AP score on coco validation set(6352), 
ref: https://github.com/ildoonet/tf-pose-estimation/blob/master/etcs/experiments.md
-----------------------
|Model/Error|Average AP 50|Average AP|
| ----|----|----|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19`|0.685686|0.304271|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21`|0.569354|0.205967|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v22`|0.656109|0.280721|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v23`|0.650248|0.274310|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v26`|0.631966|0.262369|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v27`|0.613742|0.255550|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v28`|0.654854|0.256111|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29`|0.614106|0.248199|
|`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v30`|-|


Test error rate on panoptic validation set(5556)
-----------------------
|Model/Error|Average Error|
| ----|----|


Test error rate on coco validation set(6352)
-----------------------
|Model/Error|Average Error|
| ----|----|




