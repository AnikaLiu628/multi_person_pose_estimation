`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21`
===

```bash
python3 train.py \
--dataset_path=/datasets/t2/data/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_train.record-00000-of-00001,/datasets/t2/data/coco/intermediate/coco_mp_keypoints_rot90fullfeature_gk3th3_train.record-00000-of-00001,/datasets/t3/data/panoptic/intermediate/panoptic_mp_keypoints_test.record-00000-of-00001 \
--validationset_path=/datasets/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_val.record-00000-of-00001 \
--output_model_path=../models/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v21 \
--pretrained_model_path=../models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt \
--model_type=MobilePaf \
--backbone=mobilenet_thin \
--loss_fn=MSE \
--layer_depth_multiplier=0.75 \
--number_keypoints=17 \
--batch_size=8 \
--pretrained_model=True \
--data_augmentation=False \
--optimizer=Adam \
--learning_rate=0.001 \
--decay_steps=1000000 \
--decay_factor=0.1 \
--training_steps=300000 \
--validation_interval=1000 \
--validation_batch_size=64 \
--ohem_top_k=8 

```
