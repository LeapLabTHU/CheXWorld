# Fine-tuning

Example finetuning command:

```bash
torchrun --nproc_per_node 1 --nnodes 1  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
train_finetune.py --dataset vindr_new --data_pct 1.0 --dataset_cat 1 --norm_type default  --model vit_base \
    --print_freq 20 --batch_size 256 --num_workers 16 --amp \
    --epochs 50 --warmup_epochs 1 --eval_freq 1 \
    --input_size 224 --resize_size 256  --aug_type jit --rot 0 --crop_type rrc --scale_min 0.4 \
    --early_stop 15 --layer_decay 0.75 --drop_path 0.6 \
    --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 --clip_grad 1 --save_mode best \
    --output_dir <OUTPUT_PATH> --use_target --pretrained <PRETRAINED_MODEL_PATH> --seed 0
```

By default, we use the **target encoder** for downstream finetuning.

Some hyperparameter recommendations:

- **Classification datasets (except CheXpert):**  LR=1e-4, weight_decay=0.05, layer_decay=0.75, drop_path=0.6
- **CheXpert dataset:** LR=1e-2, weight_decay=0.4, layer_decay=0.55, drop_path=0.1, epoch=1
- **Segmentation datasets:** patch_size=8, layer_decay=0.75, drop_path=0.1

