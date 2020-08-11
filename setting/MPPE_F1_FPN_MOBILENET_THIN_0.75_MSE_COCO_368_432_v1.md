`MPPE_F1_FPN_MOBILENET_THIN_0.75_MSE_COCO_368_432_v1`
===

```bash

python3 train.py 
--dataset_path=/datasets/t3/data/coco/intermediate/coco_mp_keypoints_outsize4_gk2th2_train.record-00000-of-00001,/datasets/t3/data/coco/intermediate/coco_mp_keypoints_rot90_outsize4_gk2th2_train.record-00000-of-00001,/datasets/t3/data/coco/intermediate/coco_mp_keypoints_rot180_outsize4_gk2th2_train.record-00000-of-00001,/datasets/t3/data/coco/intermediate/coco_mp_keypoints_rot270_outsize4_gk2th2_train.record-00000-of-00001,/datasets/t3/data/coco/intermediate/coco_mp_keypoints_flip_outsize4_gk2th2_train.record-00000-of-00001 \
--validationset_path=/datasets/t3/data/coco/intermediate/coco_mp_keypoints_outsize4_gk2th2_val.record-00000-of-00001 \
--output_model_path=../models/MPPE_F1_FPN_MOBILENET_THIN_0.75_MSE_COCO_368_432_v1 \
--pretrained_model_path=../models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt \
--model_type=MobilePaf_ou4 \
--backbone=mobilenet_thin_FPN \
--loss_fn=MSE \
--layer_depth_multiplier=0.75 \
--number_keypoints=17 \
--batch_size=8 \
--pretrained_model=True \
--data_augmentation=False \
--optimizer=Adam \
--learning_rate=0.001 \
--decay_steps=10000000 \
--decay_factor=0.1 \
--training_steps=1800000 \
--validation_interval=1000 \
--validation_batch_size=64 \
--ohem_top_k=8 

```