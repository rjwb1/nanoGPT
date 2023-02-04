#python nanogpt/train.py config/train_forex.py

torchrun --standalone --nproc_per_node=2 nanogpt/train.py config/train_forex.py