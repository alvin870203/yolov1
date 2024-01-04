"""
Training script for a classifier.
To run, example:
$ python train_classify.py config/train_extraction_imagenet2012.py --n_worker=1
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import v2  # not used, v2 is somehow slower in this case
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Default config values
# Task related
task_name = 'classify'
eval_only = False  # if True, script exits right after the first eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'pretrained(FUTURE)'
resume_dir = None  # only used when init_from='resume'
# Data related
dataset_name = 'imagenet2012'
img_h = 224
img_w = 224
n_class = 1000
# Transform related
scale_min = 0.25
scale_max = 1.0
imgs_mean = (0.485, 0.456, 0.406)
imgs_std = (0.229, 0.224, 0.225)
# Model related
model_name = 'extraction'
# Train related
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_iters = 100000  # total number of training iterations
# Optimizer related
learning_rate = 1e-3  # max learning rate
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = True  # whether to decay the learning rate
warmup_iters = 5000  # how many steps to warm up for
lr_decay_iters = 100000  # should be ~= max_iters
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # whether to use fused optimizer kernel
# Eval related
eval_interval = 100  # keep frequent if we'll overfit
eval_iters = 200  # use more iterations to get good estimate
# Log related
timestamp = '00000000-000000'
out_dir = f'out/extraction_imagenet2012/{timestamp}'
wandb_log = False  # disabled by default
wandb_project = 'imagenet2012'
wandb_run_name = f'extraction_{timestamp}'
log_interval = 50  # don't print too often
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


# Dataloader
data_dir = os.path.join('data', dataset_name)
match dataset_name:
    case 'imagenet2012':
        dataset_train = torchvision.datasets.ImageNet(
            data_dir, split='train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(size=(img_h, img_w), scale=(scale_min, scale_max), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=imgs_mean, std=imgs_std)
            ])
        )
        dataset_val = torchvision.datasets.ImageNet(
            data_dir, split='val',
            transform=transforms.Compose([
                transforms.Resize(size=(img_h, img_w)),
                transforms.CenterCrop(size=(img_h, img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imgs_mean, std=imgs_std)
            ])
        )
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                      num_workers=n_worker, pin_memory=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,  # shuffle since eval on only partial data
                                    num_workers=n_worker, pin_memory=True)
        print(f"train dataset: {len(dataset_train)} samples, {len(dataloader_train)} batches")
        print(f"val dataset: {len(dataset_val)} samples, {len(dataloader_val)} batches")
    case _:
        raise ValueError(f"dataset_name: {dataset_name} not supported")

class BatchGetter:
    assert len(dataloader_train) >= eval_iters, f"Not enough batches in train loader for eval."
    assert len(dataloader_val) >= eval_iters, f"Not enough batches in val loader for eval."
    dataiter = {'train': iter(dataloader_train), 'val': iter(dataloader_val)}

    @classmethod
    def get_batch(cls, split):
        try:
            X, Y = next(cls.dataiter[split])
        except StopIteration:
            cls.dataiter[split] = iter(dataloader_train) if split == 'train' else iter(dataloader_val)
            X, Y = next(cls.dataiter[split])

        if device_type == 'cuda':
            # X, Y is pinned in dataloader, which allows us to move them to GPU asynchronously (non_blocking=True)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)

        return X, Y


# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# Model init
if init_from == 'scratch':
    # init a new model from scratch
    print(f"Initializing a new {model_name} model from scratch")
elif init_from == 'resume':
    print(f"Resuming training {model_name} from {resume_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(resume_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    assert model_name == checkpoint['config']['model_name'], "model_name mismatch"
    assert dataset_name == checkpoint['config']['dataset_name'], "dataset_name mismatch"
else:
    pass  # FUTURE: init from pretrained

match model_name:
    case 'extraction':
        from model.extraction import ExtractionConfig, Extraction
        model_args = dict(
            img_h=img_h,
            img_w=img_w,
            n_class=n_class)  # start with model_args from command line
        if init_from == 'resume':
            # Force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['img_h', 'img_w', 'n_class']:
                model_args[k] = checkpoint_model_args[k]
        # Create the model
        model_config = ExtractionConfig(**model_args)
        model = Extraction(model_config)
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

model.to(device)


# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, use_fused)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory


# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = BatchGetter.get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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
X, Y = BatchGetter.get_batch('train')  # fetch the very first batch
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
        losses = estimate_loss()
        tqdm.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                tqdm.write(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))  # TODO: save top k or all checkpoints

    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = BatchGetter.get_batch('train')

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
