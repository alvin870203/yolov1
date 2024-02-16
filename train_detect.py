"""
Training script for a detector.
To run, example:
$ python train_detect.py config/train_yolov1_voc.py --n_worker=1
"""


import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.ops import box_convert
from torchvision.datasets import wrap_dataset_for_transforms_v2
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision


# -----------------------------------------------------------------------------
# Default config values
# Task related
task_name = 'detect'
eval_only = False  # if True, script exits right after the first eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'backbone' or 'pretrained(FUTURE)'
from_ckpt = ''  # only used when init_from='resume' or 'backbone'
# Data related
dataset_name = 'voc'
img_h = 448
img_w = 448
n_class = 20
# Transform related
scale_min = 0.44
scale_max = 1.44
aspect_min = 0.5
aspect_max = 2.0
brightness = 0.5
contrast = 0.0
saturation = 0.5
hue = 0.1
imgs_mean = (0.485, 0.456, 0.406)
imgs_std = (0.229, 0.224, 0.225)
# Model related
model_name = 'yolov1'
n_bbox_per_cell = 2  # B in the paper
n_grid_h = 7  # S in the paper
n_grid_w = 7  # S in the paper
reduce_head_stride = False  # only stride 1 in the head, as apposed to stride 2 in the paper
sigmoid_conf = False  # sigmoid the confidence score in the head
# Loss related
lambda_noobj = 0.5
lambda_obj = 1.0
lambda_class = 1.0
lambda_coord = 5.0
match_iou_type = 'default'  # 'default' or 'distance'
rescore = False  # whether to take the predicted iou as the target for the confidence score instead of 1.0
# Train related
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_iters = 100  # total number of training iterations
# Optimizer related
optimizer_type = 'adamw'  # 'adamw' or 'sgd'
learning_rate = 1e-3  # max learning rate
weight_decay = 1e-2
beta1 = 0.9  # beta1 for adamw, momentum for sgd
beta2 = 0.999
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = True  # whether to decay the learning rate
warmup_iters = 10  # how many steps to warm up for
lr_decay_iters = 100  # should be ~= max_iters
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # whether to use fused optimizer kernel
# Eval related
eval_interval = 5  # keep frequent if we'll overfit
eval_iters = 2  # use more iterations to get good estimate
prob_thresh = 0.001  # threshold for predicted class-specific confidence score (= obj_prob * class_prob)
iou_thresh = 0.5  # for NMS
# Log related
timestamp = '00000000-000000'
out_dir = f'out/yolov1_voc/{timestamp}'
wandb_log = False  # disabled by default
wandb_project = 'voc'
wandb_run_name = f'yolov1_{timestamp}'
log_interval = 2  # don't print too often
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
# System related
device = 'cuda'  # example: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
n_worker = 0
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# Various inits, derived attributes, I/O setup
imgs_per_iter = gradient_accumulation_steps * batch_size
print(f"imgs_per_iter will be: {imgs_per_iter}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# Collate function for object detection
def collate_fn(batch):
    xs, ys, y_supps = [], [], []
    for x, y, y_supp in batch:
        xs.append(x)
        ys.append(y)
        y_supps.append(y_supp)
    return torch.stack(xs), torch.stack(ys), y_supps

# Dataloader
data_dir = os.path.join('data', dataset_name)
match dataset_name:
    case 'voc':
        from torchvision.datasets import VOCDetection
        class Voc2Yolov1(nn.Module):
            def forward(self, x, y_voc):
                boxes_yolov1 = y_voc['boxes'].clone()
                # Transform the bounding boxes from xyxy to cxcywh normalized by the image size
                boxes_yolov1 = box_convert(boxes_yolov1, in_fmt='xyxy', out_fmt='cxcywh')
                boxes_yolov1[:, [0, 2]] /= img_w
                boxes_yolov1[:, [1, 3]] /= img_h
                # Randomly shuffle the bounding boxes and labels, since only one object can be assigned to a grid cell
                idx = torch.randperm(len(boxes_yolov1))
                y_voc['boxes'] = y_voc['boxes'][idx]
                boxes_yolov1 = boxes_yolov1[idx]
                y_voc['labels'] = y_voc['labels'][idx] - 1  # remove background class
                y_yolov1 = torch.zeros((n_grid_h, n_grid_w, 1 + 1 + 4), dtype=torch.float32)
                cx_yolov1, cy_yolov1, w_yolov1, h_yolov1 = torch.unbind(boxes_yolov1, dim=1)
                grid_x = torch.clamp_max(torch.floor(cx_yolov1 * n_grid_w), (n_grid_w - 1)).to(torch.int64)
                grid_y = torch.clamp_max(torch.floor(cy_yolov1 * n_grid_h), (n_grid_h - 1)).to(torch.int64)
                y_yolov1[grid_y, grid_x, 0] = 1.0  # set the is_obj to 1.0
                y_yolov1[grid_y, grid_x, 1] = y_voc['labels'].to(torch.float32)  # set the class index to label
                # Set the bbox coordinates, x,y are normalized by the grid size
                y_yolov1[grid_y, grid_x, 2] = cx_yolov1 * n_grid_w - grid_x
                y_yolov1[grid_y, grid_x, 3] = cy_yolov1 * n_grid_h - grid_y
                y_yolov1[grid_y, grid_x, 4] = torch.sqrt(w_yolov1)
                y_yolov1[grid_y, grid_x, 5] = torch.sqrt(h_yolov1)
                return x, y_yolov1, y_voc

        transforms_train = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            v2.RandomZoomOut(fill={tv_tensors.Image: (0, 0, 0), "others": 0}, side_range=(1.0, scale_max)),
            v2.RandomIoUCrop(min_scale=scale_min, max_scale=1.0, min_aspect_ratio=aspect_min, max_aspect_ratio=aspect_max),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize(size=(img_h, img_w), antialias=True),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            Voc2Yolov1(),
        ])
        transforms_val = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(img_h, img_w), antialias=True),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            Voc2Yolov1(),
        ])
        dataset_2007_trainval = VOCDetection(data_dir, year='2007', image_set='trainval', transforms=transforms_train)
        dataset_2012_trainval = VOCDetection(data_dir, year='2012', image_set='trainval', transforms=transforms_train)
        dataset_2007_trainval = wrap_dataset_for_transforms_v2(dataset_2007_trainval, target_keys=['boxes', 'labels'])
        dataset_2012_trainval = wrap_dataset_for_transforms_v2(dataset_2012_trainval, target_keys=['boxes', 'labels'])
        dataset_train = ConcatDataset([dataset_2007_trainval, dataset_2012_trainval])
        dataset_val = VOCDetection(data_dir, year='2007', image_set='test', transforms=transforms_val)
        dataset_val = wrap_dataset_for_transforms_v2(dataset_val, target_keys=['boxes', 'labels'])
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                      num_workers=n_worker, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,  # shuffle since eval on only partial data
                                    num_workers=n_worker, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        print(f"train dataset: {len(dataset_train)} samples, {len(dataloader_train)} batches")
        print(f"val dataset: {len(dataset_val)} samples, {len(dataloader_val)} batches")
    case _:
        raise ValueError(f"dataset_name: {dataset_name} not supported")

class BatchGetter:  # FIXME: why loss explode at specific epochs? but total avg loss is actually normal? because momentum of adamw?
    assert len(dataloader_train) >= eval_iters, f"Not enough batches in train loader for eval."
    assert len(dataloader_val) >= eval_iters, f"Not enough batches in val loader for eval."
    dataiter = {'train': iter(dataloader_train), 'val': iter(dataloader_val)}

    @classmethod
    def get_batch(cls, split):
        try:
            X, Y, Y_supp = next(cls.dataiter[split])
        except StopIteration:
            cls.dataiter[split] = iter(dataloader_train) if split == 'train' else iter(dataloader_val)
            X, Y, Y_supp = next(cls.dataiter[split])

        if device_type == 'cuda':
            # X, Y is pinned in dataloader, which allows us to move them to GPU asynchronously (non_blocking=True)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)

        return X, Y, Y_supp


# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# Model init
if init_from == 'scratch':
    # Init a new model from scratch
    print(f"Initializing a new {model_name} model from scratch")
elif init_from == 'resume':
    print(f"Resuming training {model_name} from {from_ckpt}")
    # Resume training from a checkpoint
    checkpoint = torch.load(from_ckpt, map_location='cpu')  # load to CPU first to avoid GPU OOM
    torch.set_rng_state(checkpoint['rng_state'].to('cpu'))
    checkpoint_model_args = checkpoint['model_args']
    assert model_name == checkpoint['config']['model_name'], "model_name mismatch"
    assert dataset_name == checkpoint['config']['dataset_name'], "dataset_name mismatch"
elif init_from == 'backbone':
    print(f"Initializing a {model_name} model with pretrained backbone weights: {from_ckpt}")
    # Init a new model with pretrained backbone weights
    checkpoint = torch.load(from_ckpt, map_location='cpu')
else:
    pass  # FUTURE: init from pretrained

match model_name:
    case 'yolov1':
        from model.yolov1 import Yolov1Config, Yolov1
        model_args = dict(
            img_h=img_h,
            img_w=img_w,
            n_class=n_class,
            n_bbox_per_cell=n_bbox_per_cell,
            n_grid_h=n_grid_h,
            n_grid_w=n_grid_w,
            lambda_noobj=lambda_noobj,
            lambda_obj=lambda_obj,
            lambda_class=lambda_class,
            lambda_coord=lambda_coord,
            prob_thresh=prob_thresh,
            iou_thresh=iou_thresh,
            match_iou_type=match_iou_type,
            rescore=rescore,
            reduce_head_stride=reduce_head_stride,
            sigmoid_conf=sigmoid_conf)  # start with model_args from command line
        if init_from == 'resume':
            # Force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['img_h', 'img_w', 'n_class', 'n_bbox_per_cell', 'n_grid_h', 'n_grid_w', 'reduce_head_stride']:
                model_args[k] = checkpoint_model_args[k]
        # Create the model
        model_config = Yolov1Config(**model_args)
        model = Yolov1(model_config)
    case _:
        raise ValueError(f"model_name: {model_name} not supported")

if init_from == 'resume':
    state_dict = checkpoint['model']
    # Fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from == 'backbone':
    state_dict = checkpoint['model']
    wanted_prefix = 'backbone.'
    for k,v in list(state_dict.items()):
        if not k.startswith(wanted_prefix):
            state_dict.pop(k)
        else:
            state_dict[k[len(wanted_prefix):]] = state_dict.pop(k)
    model.backbone.load_state_dict(state_dict)

model.to(device)


# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# Optimizer
optimizer = model.configure_optimizers(optimizer_type, weight_decay, learning_rate, (beta1, beta2), device_type, use_fused)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory


# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.inference_mode()
def estimate_loss():
    out_losses, out_map50 = {}, {}
    out_losses_noobj, out_losses_obj, out_losses_class, out_losses_coord = {}, {}, {}, {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters * gradient_accumulation_steps)
        losses_noobj = torch.zeros(eval_iters * gradient_accumulation_steps)
        losses_obj = torch.zeros(eval_iters * gradient_accumulation_steps)
        losses_class = torch.zeros(eval_iters * gradient_accumulation_steps)
        losses_coord = torch.zeros(eval_iters * gradient_accumulation_steps)
        metric = MeanAveragePrecision(iou_type='bbox')
        metric.warn_on_many_detections = False
        for k in range(eval_iters * gradient_accumulation_steps):
            X, Y, Y_supp = BatchGetter.get_batch(split)
            with ctx:
                logits, loss, loss_noobj, loss_obj, loss_class, loss_coord = model(X, Y)
            losses[k] = loss.item()
            losses_noobj[k] = loss_noobj.item()
            losses_obj[k] = loss_obj.item()
            losses_class[k] = loss_class.item()
            losses_coord[k] = loss_coord.item()
            preds_for_map, targets_for_map = model.postprocess_for_map(logits, Y_supp)
            metric.update(preds_for_map, targets_for_map)
        map50 = metric.compute()['map_50'].mul(100.0).item()
        out_losses[split] = losses.mean()
        out_losses_noobj[split] = losses_noobj.mean()
        out_losses_obj[split] = losses_obj.mean()
        out_losses_class[split] = losses_class.mean()
        out_losses_coord[split] = losses_coord.mean()
        out_map50[split] = map50
    model.train()
    return out_losses, out_losses_noobj, out_losses_obj, out_losses_class, out_losses_coord, out_map50


# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# Logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# Training loop
X, Y, Y_supp = BatchGetter.get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
pbar = tqdm(total=max_iters, initial=iter_num, dynamic_ncols=True)

while True:

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses, losses_noobj, losses_obj, losses_class, losses_coord, map50 = estimate_loss()
        tqdm.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val map50 {map50['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "train/loss_noobj": losses_noobj['train'],
                "train/loss_obj": losses_obj['train'],
                "train/loss_class": losses_class['train'],
                "train/loss_coord": losses_coord['train'],
                "train/map50": map50['train'],
                "val/loss": losses['val'],
                "val/loss_noobj": losses_noobj['val'],
                "val/loss_obj": losses_obj['val'],
                "val/loss_class": losses_class['val'],
                "val/loss_coord": losses_coord['val'],
                "val/map50": map50['val'],
                "lr": lr
            })

        is_last_eval = iter_num + eval_interval > max_iters
        if losses['val'] < best_val_loss or always_save_checkpoint or is_last_eval:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'rng_state': torch.get_rng_state()
            }
            if iter_num > 0:
                tqdm.write(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))  # TODO: save top k checkpoints
            if is_last_eval and not always_save_checkpoint:
                tqdm.write(f"saving last checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_last.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, _, _, _, _ = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, Y_supp = BatchGetter.get_batch('train')

        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # Clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # Get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        tqdm.write(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1
    pbar.update(1)

    # Termination conditions
    if iter_num > max_iters:
        break
