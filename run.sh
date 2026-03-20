torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
train.py   --data_dir /data_path       --dino_size l   --dino_ckpt web_pth/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth  --repo_dir ./dinov3-main      --epochs 50   --batch_size 32   --lr 2e-4  --backbone_lr 3e-5  --out_dir /output

