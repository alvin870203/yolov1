# Config for training Extraction model on ImageNet2012 dataset for image classification as the backbone for YOLOv1
import time

# Task related
task_name = 'classify'

# Data related
dataset_name = 'imagenet2012'
img_h = 224
img_w = 224
n_class = 1000

# Transform related
max_crop = 320

# Model related
model_name = 'extraction'

# Train related
# the number of examples per iter:
# 1,024 batch_size * 1 grad_accum = 1,024 imgs/iter
# imagenet2012 train set has 1,281,167 imgs, so 1 epoch ~= 1,251 iters
gradient_accumulation_steps = 1
batch_size = 1024  # TODO: change after complete model
max_iters = 1600000  # TODO

# Optimizer related
learning_rate = 1e-1  # max learning rate
weight_decay = 5e-4
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = True  # whether to decay the learning rate
warmup_iters = 10000  # how many steps to warm up for  # TODO
lr_decay_iters = 1600000  # should be ~= max_iters  # TODO
min_lr = 1e-2  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# imagenet2012 val set has 50,000 imgs, so 1 epoch ~= 48 iters
eval_interval = 1000  # keep frequent if we'll overfit  # TODO
eval_iters = 2  # use more iterations to get good estimate  # TODO

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/extraction_imagenet2012/{timestamp}'
wandb_log = True
wandb_project = 'imagenet2012'
wandb_run_name = f'extraction_{timestamp}'
log_interval = 200  # don't print too often  # TODO
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 3
