# Config for training YOLOv1 model on Pascal VOC 2007&2012 Detection dataset for object detection
# Train on VOC 2007 trainval and 2012 trainval, and evaluate on VOC 2007 test
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/extraction_imagenet2012/20240128-110502/ckpt.pt'

# Data related
dataset_name = 'voc'
img_h = 448
img_w = 448
n_class = 20

# Transform related
scale_min = 0.44  # not very sure how to set this to have the same effect as the darknet implementation
scale_max = 1.44  # not very sure how to set this to have the same effect as the darknet implementation
aspect_min = 0.5  # default of torchvision
aspect_max = 2.0  # default of torchvision
brightness = 0.5  # not very sure how to set this to have the same effect as the darknet implementation
contrast = 0.0  # not very sure how to set this to have the same effect as the darknet implementation
saturation = 0.5  # not very sure how to set this to have the same effect as the darknet implementation
hue = 0.1  # not very sure how to set this to have the same effect as the darknet implementation
imgs_mean = [0.0, 0.0, 0.0]  # no normalization for yolo
imgs_std = [1.0, 1.0, 1.0]  # no normalization for yolo

# Model related
model_name = 'yolov1'
n_bbox_per_cell = 3
n_grid_h = 7
n_grid_w = 7

# Loss related
lambda_coord = 5.0
lambda_noobj = 0.5
match_iou_type = 'distance'
rescore = True

# Train related
# the number of examples per iter:
# 128 batch_size * 1 grad_accum = 128 imgs/iter
# voc train set has 16,551 imgs, so 1 epoch ~= 129 iters
gradient_accumulation_steps = 1
batch_size = 128  # filled up the gpu memory on my machine
max_iters = 120000  # 20,000 iters * 128 batch_size = 2,560,000 imgs, same as the darknet implementation 40,000 iters * 64 batch_size = 2,560,000 imgs
                    # additional 100,000 to finish in 1 day on my machine

# Optimizer related
optimizer_type = 'adamw'
learning_rate = 1e-4  # a bit smaller than the darknet implementation to prevent divergence at the beginning
weight_decay = 5e-4
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = True  # whether to decay the learning rate
warmup_iters = 650  # warmup 5 epochs
lr_decay_iters = 120000  # should be ~= max_iters
min_lr = 1e-5  # minimum learning rate, should be ~= learning_rate/10, but set to the same as the darknet implementation
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# voc val set has 4,952 imgs, so 1 epoch ~= 38 iters
eval_interval = 650  # keep frequent if we'll overfit, but don't too frequent to get stable results
eval_iters = 16  # use more iterations to get good estimate
prob_thresh = 0.001
iou_thresh = 0.5

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/yolov1_voc/{timestamp}'
wandb_log = True
wandb_project = 'voc'
wandb_run_name = f'yolov1_{timestamp}'
log_interval = 20  # don't print too often
always_save_checkpoint = True  # only save when val improves if we expect overfit

# System related
compile = False
n_worker = 4
