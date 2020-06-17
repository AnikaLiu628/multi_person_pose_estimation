`MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v18`
===

```bash
python3 train.py 
--dataset_path=/datasets/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_train.record-00000-of-00001 \
--validationset_path=/datasets/coco/intermediate/coco_mp_keypoints_fullfeature_gk3th3_val.record-00000-of-00001 \
--output_model_path=../models/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v18 \
--pretrained_model_path=../models/MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v17/model.ckpt-137000 \
--backbone=mobilenet_thin \
--pretrained_model=True \
--learning_rate=0.001 \
--decay_steps=1000000 \
--validation_batch_size=64 \
--batch_size=8

```