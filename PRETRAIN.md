# Pre-training

Example command to train a ViT-Base on MIMIC+NIH+CheXpert, using eight RTX4090s (8x24GB VRAMs).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 --nnodes 1  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
    train_jepa.py --ssl_type iwm_dual_easy \
    --dataset mimic_nih_chex --data_postfix _proc --data_pct 1.0 --norm_type default --model vit_base \
    --print_freq 50 --batch_size 128 --accum_iter 2 --num_workers 16 --amp \
    --epochs 300 --warmup_epochs 40 --eval_freq 20 \
    --input_size 224 --resize_size 224  --aug_type jit --crop_type rrc --scale_min 0.3 \
    --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --weight_decay_end 0.05 --clip_grad 1 --ema 0.996 --loss_type l2 \
    --pred_emb_dim 384 --pred_depth 6 --mask_type multi_multiblock --enc_mask_scale 0.75 1.0  --ipe_scale 1.25 --mask_merge \
     --extra_mean --extra_global_scale 0.3 1.0 \
    --output_dir <PATH_TO_OUTPUT> --exp_name <EXP_NAME> --seed 0
```

