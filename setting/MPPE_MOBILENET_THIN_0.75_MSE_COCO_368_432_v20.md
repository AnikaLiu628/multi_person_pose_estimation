`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v20`
===

```bash

python3 train.py 
--dataset_path=/datasets/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_train.record-00000-of-00001,/datasets/coco/intermediate/coco_mp_keypoints_rot90fullfeature_gk3th3_train.record-00000-of-00001,coco_mp_keypoints_flipfullfeature_gk3th3_train.record-00000-of-00001 \
--validationset_path=/datasets/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_val.record-00000-of-00001 \
--output_model_path=../models/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v20 \
--pretrained_model_path=../models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt \
--backbone=mobilenet_thin \
--pretrained_model=True \
--learning_rate=0.001 \
--decay_steps=1000000 \
--validation_batch_size=64 \
--batch_size=8 

```