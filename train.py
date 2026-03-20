# train.py

import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import argparse
import csv
import warnings
import datetime


class DiceLoss(nn.Module):
    def __init__(self, n_classes=3, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        
        loss = 0.0
        for c in range(self.n_classes):
            intersection = (probs[:, c] * target_one_hot[:, c]).sum(dim=(1, 2))
            union = probs[:, c].sum(dim=(1, 2)) + target_one_hot[:, c].sum(dim=(1, 2))
            
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            loss += (1 - dice_score).mean()
            
        return loss / self.n_classes

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        class_weights = torch.tensor([1.0, 3.0, 5.0])
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        
        self.dice = DiceLoss(n_classes=3)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
#dataset
class ImagesMasksPatches(Dataset):
    def __init__(self, root, split="train", train_ratio=0.8, crop_size=512, seed=42, transform=None):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.transform = transform
        self.crop_size = crop_size if crop_size and crop_size > 0 else None

        img_files = sorted(glob(os.path.join(self.img_dir, "*.png")) + glob(os.path.join(self.img_dir, "*.jpg")) + glob(os.path.join(self.img_dir, "*.jpeg")))
        mask_files = sorted(glob(os.path.join(self.mask_dir, "*.png")) + glob(os.path.join(self.mask_dir, "*.jpg")) + glob(os.path.join(self.mask_dir, "*.jpeg")))

        img_map = {os.path.basename(p): p for p in img_files}
        mask_map = {os.path.basename(p): p for p in mask_files}
        common = sorted(set(img_map.keys()) & set(mask_map.keys()))
        if len(common) == 0:
            raise RuntimeError(f"No matching image/mask basenames found in {self.img_dir} and {self.mask_dir}")

        paired = [(img_map[n], mask_map[n]) for n in common]

        items = []
        if self.crop_size is None:
            for ip, mp in paired:
                im = cv2.imread(ip, cv2.IMREAD_COLOR)
                if im is None: continue
                h, w = im.shape[:2]
                basename = os.path.basename(ip)
                items.append((ip, mp, 0, 0, None, basename, (h,w)))
        else:
            ps = self.crop_size
            for ip, mp in paired:
                im = cv2.imread(ip, cv2.IMREAD_COLOR)
                if im is None: continue
                h, w = im.shape[:2]
                nh = ((h + ps - 1) // ps) * ps
                nw = ((w + ps - 1) // ps) * ps
                basename = os.path.basename(ip)
                for y in range(0, nh, ps):
                    for x in range(0, nw, ps):
                        items.append((ip, mp, x, y, ps, basename, (h,w)))

        rng = list(range(len(items)))
        random.Random(seed).shuffle(rng)
        split_idx = int(train_ratio * len(rng))
        if split == "train":
            sel = rng[:split_idx]
        else:
            sel = rng[split_idx:]
        self.items = [items[i] for i in sel]
        print(f"[Dataset] found {len(common)} paired files -> items {len(self.items)} (split={split}, crop_size={self.crop_size})")

    def __len__(self):
        return len(self.items)

    def _read_and_patch(self, ip, mp, x, y, ps):
        im = cv2.imread(ip, cv2.IMREAD_COLOR)
        mask = cv2.imread(mp, cv2.IMREAD_COLOR)
        if im is None: raise RuntimeError(f"Failed to read image {ip}")
        if mask is None: raise RuntimeError(f"Failed to read mask {mp}")
        if ps is None: return im, mask
        h, w = im.shape[:2]
        pad_h = max(0, (y + ps) - h)
        pad_w = max(0, (x + ps) - w)
        if pad_h > 0 or pad_w > 0:
            im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
        patch_im = im[y:y+ps, x:x+ps]
        patch_mask = mask[y:y+ps, x:x+ps]
        return patch_im, patch_mask

    def __getitem__(self, idx):
        ip, mp, x, y, ps, basename, (orig_h, orig_w) = self.items[idx]
        im_bgr, mask_bgr = self._read_and_patch(ip, mp, x, y, ps)

        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        h, w = mask_rgb.shape[:2]
        label = np.zeros((h, w), dtype=np.uint8)
        white_mask = (mask_rgb[:,:,0] == 255) & (mask_rgb[:,:,1] == 255) & (mask_rgb[:,:,2] == 255)
        yellow_mask = (mask_rgb[:,:,0] == 255) & (mask_rgb[:,:,1] == 255) & (mask_rgb[:,:,2] == 0)
        label[white_mask] = 1
        label[yellow_mask] = 2

        if self.transform is not None:
            img_t, label_t = self.transform(im_rgb, label)
        else:
            img_t = torch.from_numpy(im_rgb).permute(2,0,1).float()
            label_t = torch.from_numpy(label).long()

        meta = {'basename': basename, 'x': int(x), 'y': int(y), 'ps': ps, 'orig_hw': (int(orig_h), int(orig_w))}
        return img_t, label_t, meta

# transform
class ResizeNormalize:
    def __init__(self, out_size=(512,512), mean=None, std=None):
        self.out_size = out_size
        self.mean = [0.430, 0.411, 0.296]
        self.std = [0.213, 0.156, 0.143]

    def __call__(self, img_np, label_np):
        H, W = self.out_size
        img = cv2.resize((img_np * 255.0).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        label = cv2.resize(label_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        img = (img - self.mean) / self.std
        img_t = torch.from_numpy(img).permute(2,0,1).float()
        label_t = torch.from_numpy(label).long()
        return img_t, label_t

def tensor_to_bgr_uint8(img_t: torch.Tensor):
    im = img_t.detach().cpu().float().numpy()
    im = (np.transpose(im, (1,2,0)) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def mask_to_color(mask_t):
    m = mask_t.detach().cpu().numpy()
    if m.ndim == 3: m = m[0]
    h, w = m.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    out[m==0] = [0,0,0]
    out[m==1] = [255,255,255]
    out[m==2] = [255,255,0]
    return out

# backbone / checkpoint helpers
def strip_prefix_state_dict(sd, prefixes=('module.', 'model.', 'backbone.')):
    new = {}
    for k,v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p): nk = nk[len(p):]; break
        new[nk] = v
    return new

def try_load_backbone_only(backbone, ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    cands = []
    if isinstance(ck, dict):
        for key in ('state_dict', 'model_state_dict', 'backbone_state_dict', 'backbone'):
            if key in ck and isinstance(ck[key], dict): cands.append(ck[key])
        if all(isinstance(v, torch.Tensor) for v in ck.values()): cands.append(ck)
    else: return False

    for cand in cands:
        try:
            backbone.load_state_dict(cand, strict=False)
            print("[backbone] loaded candidate permissively")
            return True
        except: pass
        try:
            backbone.load_state_dict(strip_prefix_state_dict(cand), strict=False)
            print("[backbone] loaded candidate after stripping prefixes")
            return True
        except: pass
    print("[backbone] could not load backbone weights from provided ckpt")
    return False

#training
def train_epoch(model, loader, optimizer, criterion, device, epoch, local_rank):
    model.train()
    total_loss = 0.0
    total_samples = 0

    current_lr = optimizer.param_groups[1]['lr']
    
    if local_rank == 0:
        pbar = tqdm(loader, desc=f"Train Ep{epoch} (LR={current_lr:.2e})")
    else:
        pbar = loader
    
    for batch in pbar:
        if len(batch) == 3: 
            imgs, masks, metas = batch
        else: 
            imgs, masks = batch
        
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        
        if local_rank == 0:
            pbar.set_postfix(loss=loss.item())

    total_loss_tensor = torch.tensor(total_loss).to(device)
    total_samples_tensor = torch.tensor(total_samples).to(device)
    
    torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
    
    avg_loss = total_loss_tensor.item() / max(1, total_samples_tensor.item())
        
    return avg_loss

@torch.no_grad()
def validate_and_iou(model, loader, device, criterion, vis_dir=None, preds_dir_epoch=None, save_vis_samples=3, local_rank=0):
    model.eval()
    total_loss = 0.0; total_samples = 0
    total_correct = 0; total_pixels = 0

    n_classes = 3
    inter = np.zeros(n_classes, dtype=np.float64)
    union = np.zeros(n_classes, dtype=np.float64)
    class_total_pixels = np.zeros(n_classes, dtype=np.int64)
    saved = 0
    
    if vis_dir: os.makedirs(vis_dir, exist_ok=True)
    if preds_dir_epoch: os.makedirs(preds_dir_epoch, exist_ok=True)
    stitch_store = {}
    warn_meta_parsing = False

    if local_rank == 0:
        pbar = tqdm(loader, desc="Validation")
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3: imgs, masks, metas = batch
        else: imgs, masks = batch
        
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, masks)
        bs = imgs.size(0)

        total_loss += float(loss.item()) * bs
        total_samples += bs

        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == masks).sum().item())
        total_pixels += int(masks.numel())

        for c in range(n_classes):
            pred_c = (preds == c); gt_c = (masks == c)
            inter_c = int((pred_c & gt_c).sum().item())
            union_c = int(((pred_c | gt_c)).sum().item())
            inter[c] += inter_c
            union[c] += union_c
            class_total_pixels[c] += int(gt_c.sum().item())

        if local_rank == 0:
            per_sample_metas = [None] * bs
            if metas is not None:
                try:
                    if isinstance(metas, dict):
                        keys = list(metas.keys())
                        for i in range(bs):
                            single = {}
                            for k in keys:
                                v = metas[k]
                                try:
                                    val = v[i]
                                    if isinstance(val, torch.Tensor): val = val.item() if val.numel()==1 else val.cpu().numpy().tolist()
                                    if isinstance(val, (bytes, bytearray)): val = val.decode('utf-8')
                                    single[k] = val
                                except: single[k] = None
                            per_sample_metas[i] = single
                    elif isinstance(metas, (list, tuple)):
                        for i in range(bs): per_sample_metas[i] = metas[i]
                except: 
                    if not warn_meta_parsing: warnings.warn("Meta parsing failed"); warn_meta_parsing=True

            for b in range(bs):
                meta = per_sample_metas[b] if per_sample_metas else None
                pred_b = preds[b].cpu()
                
                if preds_dir_epoch:
                    fname = os.path.join(preds_dir_epoch, f"pred_{batch_idx:04d}_{b}.png")
                    col = mask_to_color(pred_b)
                    cv2.imwrite(fname, cv2.cvtColor(col, cv2.COLOR_RGB2BGR))

                if vis_dir and saved < save_vis_samples:
                    im_bgr = tensor_to_bgr_uint8(imgs[b].cpu())
                    pred_color = mask_to_color(pred_b)
                    gt_color = mask_to_color(masks[b].cpu())
                    base = os.path.join(vis_dir, f"val_{batch_idx:04d}_{b}")
                    cv2.imwrite(base + "_img.png", im_bgr)
                    cv2.imwrite(base + "_pred.png", cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(base + "_gt.png", cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))
                    saved += 1

                if meta is None: continue
                try:
                    basename = meta.get('basename', None)
                    x = int(meta.get('x', 0)); y = int(meta.get('y', 0))
                    ps = meta.get('ps', None)
                    orig_hw = meta.get('orig_hw', None)
                    if basename is None or orig_hw is None: continue
                    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])

                    if basename not in stitch_store:
                        stitch_store[basename] = {'accum': np.zeros((orig_h, orig_w), dtype=np.float32), 'count': np.zeros((orig_h, orig_w), dtype=np.float32)}

                    pred_np = pred_b.numpy().astype(np.uint8)
                    ph, pw = pred_np.shape
                    x1 = min(x + (ps if ps else pw), orig_w)
                    y1 = min(y + (ps if ps else ph), orig_h)
                    tx1 = x1 - x; ty1 = y1 - y
                    if tx1 <= 0 or ty1 <= 0: continue
                    
                    stitch_store[basename]['accum'][y:y1, x:x1] += pred_np[0:ty1, 0:tx1].astype(np.float32)
                    stitch_store[basename]['count'][y:y1, x:x1] += 1.0
                except: continue

    if local_rank == 0:
        for basename, data in stitch_store.items():
            acc = data['accum']; cnt = data['count']
            stitched = np.zeros_like(acc, dtype=np.uint8)
            nonzero = cnt > 0
            if nonzero.any():
                stitched = np.rint(acc[nonzero] / cnt[nonzero]).astype(np.uint8)
                full_stitched = np.zeros_like(acc, dtype=np.uint8)
                full_stitched[nonzero] = stitched
                stitched = full_stitched
                
            col = np.zeros((stitched.shape[0], stitched.shape[1], 3), dtype=np.uint8)
            col[stitched==1] = [255,255,255]
            col[stitched==2] = [255,255,0]
            if preds_dir_epoch:
                outp = os.path.join(preds_dir_epoch, f"stitched_{basename}")
                if not outp.endswith(".png"): outp += ".png"
                cv2.imwrite(outp, cv2.cvtColor(col, cv2.COLOR_RGB2BGR))

    avg_loss = total_loss / max(1, total_samples)
    overall_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
    per_class_iou = [inter[c]/(union[c]+1e-6) for c in range(n_classes)]
    mean_iou = float(np.mean(per_class_iou))
    per_class_acc = [(inter[c]/class_total_pixels[c]) if class_total_pixels[c]>0 else 0.0 for c in range(n_classes)]

    return avg_loss, overall_acc, per_class_acc, per_class_iou, mean_iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--repo_dir", default="./dinov3")
    parser.add_argument("--dino_ckpt", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dino_size", choices=["b","s","l"], default="l")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Learning rate for Backbone")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None, help="root output directory")
    parser.add_argument("--dist_url", type=str, default="env://", help="Distributed init URL")
    
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        args.lr = args.lr * world_size
        args.backbone_lr = args.backbone_lr * world_size
        if local_rank == 0:
            print(f"[Auto-Scale] Linear Scaling Rule Applied:")
            print(f"  - Effective Batch Size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")
            print(f"  - New Head LR: {args.lr}")
            print(f"  - New Backbone LR: {args.backbone_lr}")


    print(f"[Rank {local_rank}] Starting...")

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=world_size,
        rank=local_rank,
        timeout=datetime.timedelta(seconds=100)
    )

    seed = args.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda", local_rank)
    print(f"[Rank {local_rank}] Using device:", device)

    if local_rank == 0:
        base_name = f"segdino_multiclass_{args.dino_size}" if args.out_dir is None else os.path.basename(args.out_dir.rstrip("/"))
        out_root = args.out_dir if args.out_dir else os.path.join("./runs", base_name)
        ckpt_dir = os.path.join(out_root, "ckpts")
        train_vis = os.path.join(out_root, "train_vis")
        val_vis = os.path.join(out_root, "val_vis")
        preds_root = os.path.join(out_root, "preds")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(train_vis, exist_ok=True)
        os.makedirs(val_vis, exist_ok=True)
        os.makedirs(preds_root, exist_ok=True)
        metrics_csv = os.path.join(out_root, "metrics.csv")
        print(f"[Rank {local_rank}] Output directories created at {out_root}")
    else:
        base_name = f"segdino_multiclass_{args.dino_size}" if args.out_dir is None else os.path.basename(args.out_dir.rstrip("/"))
        out_root = args.out_dir if args.out_dir else os.path.join("./runs", base_name)
        ckpt_dir = os.path.join(out_root, "ckpts")
        preds_root = os.path.join(out_root, "preds")
        metrics_csv = os.path.join(out_root, "metrics.csv")

    torch.distributed.barrier()
    print(f"[Rank {local_rank}] After barrier 1")

    print(f"[Rank {local_rank}] Loading DINOv3 skeleton...")
    try:
        if args.dino_size == "b":
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', pretrained=False)
        elif args.dino_size == "s":
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', pretrained=False)
        else:
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vitl16', source='local', pretrained=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load dinov3 skeleton from repo_dir={args.repo_dir}: {e}")

    if args.dino_ckpt and os.path.exists(args.dino_ckpt):
        try:
            ok = try_load_backbone_only(backbone, args.dino_ckpt)
            if not ok:
                if local_rank == 0:
                    print("[warn] dino_ckpt could not be fully applied; continuing with skeleton.")
        except Exception as e:
            if local_rank == 0:
                print("[warn] try_load_backbone_only failed:", e)
# decoder
    from dblock_mamba import dual_decoder
    model = dual_decoder(nclass=3, backbone=backbone)
    model = model.to(device)
    
    if world_size > 1:
        import torch.nn as nn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if local_rank == 0:
            print("[Info] SyncBatchNorm enabled! This is critical for DDP segmentation.")
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    criterion = CombinedLoss().to(device)
    print(f"[Rank {local_rank}] Using Differential Learning Rates. Backbone: {args.backbone_lr}, Head: {args.lr}")

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    param_groups = [
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.lr},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=0.05)
    warmup_epochs = 5

    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )

    scheduler_main = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - warmup_epochs, 
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_main],
        milestones=[warmup_epochs]
    )

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        if local_rank == 0:
            print(f"[resume] loading from {args.resume}")

        checkpoint = torch.load(args.resume, map_location='cpu')

        model.module.load_state_dict(checkpoint['state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        
        if local_rank == 0:
            print(f"[resume] continue from epoch {start_epoch}")

    transform = ResizeNormalize(out_size=(args.crop_size, args.crop_size)) if args.crop_size else None

    train_ds = ImagesMasksPatches(args.data_dir, split="train", train_ratio=0.8, 
                                 crop_size=args.crop_size, seed=args.seed, transform=transform)

    val_ds = ImagesMasksPatches(args.data_dir, split="val", train_ratio=0.8,
                               crop_size=args.crop_size, seed=args.seed, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=args.seed
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    if local_rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=min(4, args.num_workers),
            pin_memory=True
        )

    if local_rank == 0 and not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch","train_loss","val_loss","val_acc","class0_acc","class1_acc","class2_acc",
                      "iou_class0","iou_class1","iou_class2","mean_iou"]
            writer.writerow(header)
    
    best_mean_iou = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        if local_rank == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, local_rank)
        
        if local_rank == 0:
            print(f" train_loss: {train_loss:.4f}")

        scheduler.step()

        if local_rank == 0 and val_loader is not None:
            preds_dir_epoch = os.path.join(preds_root, f"epoch_{epoch:03d}")
            os.makedirs(preds_dir_epoch, exist_ok=True)
            
            val_loss, val_acc, class_acc, per_class_iou, mean_iou = validate_and_iou(
                model.module,
                val_loader, 
                device, 
                criterion, 
                vis_dir=val_vis, 
                preds_dir_epoch=preds_dir_epoch,
                save_vis_samples=3,
                local_rank=local_rank
            )
            
            print(f" val_loss={val_loss:.4f} val_acc={val_acc:.4f} mean_iou={mean_iou:.4f}")

            latest_p = os.path.join(ckpt_dir, "latest.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "mean_iou": mean_iou
            }, latest_p)
            
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                best_p = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_miou{mean_iou:.4f}.pth")
                torch.save(model.module.state_dict(), best_p)
                print(f"[save] new best by mean_iou: {best_p}")

            with open(metrics_csv, "a", newline="") as f:
                writer = csv.writer(f)
                row = [epoch, train_loss, val_loss, val_acc] + class_acc + per_class_iou + [mean_iou]
                writer.writerow(row)

        torch.distributed.barrier()
    
    if local_rank == 0:
        print(f"Training finished. best_mean_iou={best_mean_iou}")
    
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()