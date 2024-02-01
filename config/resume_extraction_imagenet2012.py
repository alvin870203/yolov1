# Config for training Extraction model on ImageNet2012 dataset for image classification as the backbone for YOLOv1
import time

# Task related
task_name = 'classify'
init_from = 'resume'
from_ckpt = 'out/extraction_imagenet2012/20240103-160048/ckpt.pt'

# Data related
dataset_name = 'imagenet2012'
img_h = 224
img_w = 224
n_class = 1000

# Transform related
scale_min = 0.25  # not very sure how to set this to have the same effect as the darknet implementation
scale_max = 1.0  # not very sure how to set this to have the same effect as the darknet implementation
imgs_mean = [0.0, 0.0, 0.0]  # no normalization for yolo
imgs_std = [1.0, 1.0, 1.0]  # no normalization for yolo

# Model related
model_name = 'extraction'

# Train related
# the number of examples per iter:
# 1,024 batch_size * 1 grad_accum = 1,024 imgs/iter
# imagenet2012 train set has 1,281,167 imgs, so 1 epoch ~= 1,251 iters
gradient_accumulation_steps = 1
batch_size = 1024  # filled up the gpu memory on my machine
max_iters = 125100+37530  # resume from 125100 and train for 30 epochs more

# Optimizer related
learning_rate = 1e-3  # max learning rate
weight_decay = 5e-4
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = True  # whether to decay the learning rate
warmup_iters = 6255  # warmup 5 epochs
lr_decay_iters = 125100  # should be ~= max_iters
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# imagenet2012 val set has 50,000 imgs, so 1 epoch ~= 48 iters
eval_interval = 2502  # keep frequent if we'll overfit
eval_iters = 2  # use more iterations to get good estimate

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/extraction_imagenet2012/{timestamp}'
wandb_log = True
wandb_project = 'imagenet2012'
wandb_run_name = f'extraction_{timestamp}'
log_interval = 200  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 4
