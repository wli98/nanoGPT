#torchrun --standalone --nproc_per_node=1 train.py config/train_custom_gpt2.py
#torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2.py
#TORCH_DISTRIBUTED_DEBUG=DETAIL 
#torchrun --standalone --nproc_per_node=1 train.py config/train_custom_gpt2_3.py
torchrun --standalone --nproc_per_node=1 train.py config/train_custom_gpt2.py

