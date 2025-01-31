import os, sys
import numpy as np
import csv 
import pickle
from scipy.stats import beta

data_file = "EURUSD_H1.csv"
data_index = 4 
vocab_size = 250
data = []
# Import Forex Data from CSV
input_file_path = os.path.join(os.path.dirname(__file__), data_file)
if not os.path.exists(input_file_path):
    print("forex csv not found: %s" % data_file)
    sys.exit()

previous_data_point = 0

with open(input_file_path, 'r') as f:
    csv_file = csv.reader(f)
    for i, row in enumerate(csv_file):
        if i == 0:
            previous_price = float(row[4])
        else:
            data.append(float(row[4]) - float(previous_price)) 
            previous_price = float(row[4])

def denseboundspace(size=30, start=0, end=9, alpha=.5):
    x = np.linspace(0, 1, size)
    return start + beta.isf(x, 2.+alpha, 2.+alpha) * (end-start)

n = len(data)

data_min = min(data)
data_max = max(data)

bins = denseboundspace(vocab_size, data_min, data_max, 0.01)

std_d = np.std(data)

print(f"data has minimum change of = {data_min:,}")
print(f"data has max change of {data_max:,}")
 
data = np.digitize(data, bins)
x = np.tan(bins)

train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = train_data
val_ids = val_data

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': None,
    'stoi': None,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)