`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29`
===

```bash
python3 train.py \
--dataset_path=/datasets/t2/data/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_train.record-00000-of-00001,/datasets/t2/data/coco/intermediate/coco_mp_keypoints_rot90fullfeature_gk3th3_train.record-00000-of-00001,/datasets/t3/data/panoptic/intermediate/panoptic_mp_keypoints_train.record-00000-of-00001,/datasets/t2/data/coco/intermediate/coco_mp_keypoints_rot270fullfeature_gk3th3_train.record-00000-of-00001,/datasets/t2/data/coco/intermediate/coco_mp_keypoints_rot180fullfeature_gk3th3_train.record-00000-of-00001 \
--validationset_path=/datasets/t2/data/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_val.record-00000-of-00001 \
--output_model_path=../models/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v29 \
--pretrained_model_path=../models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt \
--backbone=mobilenet_thin \
--pretrained_model=True \
--learning_rate=0.001 \
--decay_steps=10000000 \
--validation_batch_size=64 \
--batch_size=8 \
--training_steps=1000000
```