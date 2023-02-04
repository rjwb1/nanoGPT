import os, sys
import numpy as np
import csv 

data_file = "EURUSD_H1.csv"
data_index = 4
data = []
# Import Forex Data from CSV
input_file_path = os.path.join(os.path.dirname(__file__), data_file)
if not os.path.exists(input_file_path):
    print("forex csv not found: %s" % data_file)
    sys.exit()

with open(input_file_path, 'r') as f:
    csv_file = csv.reader(f)
    for i, row in enumerate(csv_file):
        data.append(row[4]) 
n = len(data)

data_min = min(data)
data_max = max(data)

train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
# enc = tiktoken.get_encoding("gpt2")
# train_ids = enc.encode_ordinary(train_data)
# val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
