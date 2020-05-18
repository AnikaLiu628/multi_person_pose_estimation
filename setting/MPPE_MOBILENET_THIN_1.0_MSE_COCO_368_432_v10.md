`MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v10`
===

```bash
python3 train.py \
--dataset_path=/datasets/coco/intermediate/coco_mp_keypoints_feature_train.record-00000-of-00001 \
--validationset_path=/datasets/coco/intermediate/coco_mp_keypoints_feature_train.record-00000-of-00001 \
--output_model_path=../models/MPPE_MOBILENET_THIN_1.0_MSE_COCO_368_432_v10 \
--model_type=MobilePifPaf \
--backbone=mobilenet_thin \
--loss_fn=MSE \
--layer_depth_multiplier=1.0 \
--number_keypoints=17 \
--batch_size=16 \
--pretrained_model=False \
--data_augmentation=False \
--optimizer=Adam \
--learning_rate=0.001 \
--decay_steps=1000000000 \
--decay_factor=0.1 \
--training_steps=300000 \
--validation_interval=1000 \
--validation_batch_size=128 \
--ohem_top_k=8 

```