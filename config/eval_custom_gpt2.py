# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'resume'
out_dir = 'ckpts'

block_size = 1024
window_training=True
attend_embed=True
cross_encode=True
y_transformer=True

n_layer=9
y_mlp=False
y_mlp_depth=3
window_size = 64
interm_layer_idx = 6
n_y_layers = 3

start = '<|endoftext|> The meaning of life'
tok_k = 10
