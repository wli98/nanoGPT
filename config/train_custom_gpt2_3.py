# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='y_6_6_1_win64'
log_grad = True
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 16
block_size = 1024
window_training=True
attend_embed=True
cross_encode=True
y_transformer=True

n_layer=6
y_mlp=False
y_mlp_depth=3
window_size = 64
interm_layer_idx = 1
n_y_layers = 6

gradient_accumulation_steps = 1

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 20
log_interval = 10

# weight decay
weight_decay = 1e-1
