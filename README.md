# A Vision-Language Foundation AI Model for Clinical Eyecare Needs: Multi-stage Evaluation in Primary and Specialty Settings

## Contents

1. Requirements
1. Finetuning of vision module
1. Pretraining of vision module
1. Finetuning of vision-language
5. Pretraining of vision-language

## Requirements

This software requires a **Linux** system: [**Ubuntu 22.04**](https://ubuntu.com/download/desktop) or  [**Ubuntu 20.04**](https://ubuntu.com/download/desktop) (other versions are not tested)   and  [**Python3.8**](https://www.python.org) (other versions are not tested).

The **Python packages** needed are listed below. They can also be found in `reqirements.txt`.

```
torch>=1.10.0
torchvision>=0.11.1
timm>=0.4.12
einops>=0.3.2
pandas>=1.3.4
albumentations>=1.1.0
wandb>=0.12.11
flash-attn==2.5.8,
ninja>=1.11.1.0
open-clip-torch>=2.11.0
```

## Finetuning of vision module

For the finetuning of vision downstream tasks:

```
cd image_module
python run_finetuning.py --config cfgs/finetune/finetune.yaml --finetune model_weights/checkpoint-199.pth --data_path path/to/images --csv_path path/to/csv --in_domains cfp --label_column DR --nb_classes 5
```

## Pretraining of vision module

For the pretraining of vision module:

```
cd image_module
python run_pretraining.py --config cfgs/pretrain/pretraining.yaml
```

## Finetuning of vision-language

For the finetuning of vision-language downstream tasks:

```
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 train/train_mem.py --model_name_or_path /EyeFM/cn_llava  --data_path /EyeFM/ours_cn_task1.json --image_folder / --tune_mm_mlp_adapter True --output_dir /EyeFM/output --vision_tower /EyeFM/transfer_image_encoder
 --mm_vision_select_layer -2 --mm_use_im_start_end True --num_train_epochs 5 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 1024 --lazy_preprocess True --gradient_checkpointing True --dataloader_num_workers 8 --report_to none --bf16 false --tf32 false
```

## Pretraining of vision-language

For the pretraining of vision-language:

```
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 train/train_mem.py --model_name_or_path /EyeFM/cn_llava  --data_path /EyeFM/ours_cn_all.json --image_folder / --tune_mm_mlp_adapter True --output_dir /EyeFM/output --vision_tower /EyeFM/transfer_image_encoder --mm_vision_select_layer -2 --mm_use_im_start_end True --num_train_epochs 5 --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 2e-3 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 1024 --lazy_preprocess True --gradient_checkpointing True --dataloader_num_workers 8 --report_to none --bf16 false --tf32 false
```
