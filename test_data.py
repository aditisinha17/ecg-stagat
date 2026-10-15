"""
Data loading validation script.
"""
import time
from ecg_stgat.data.ptbxl_dataset import get_ptbxl_dataloaders

t0 = time.time()
train_l, val_l, test_l, pos_w, df = get_ptbxl_dataloaders(
    data_dir="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
t1 = time.time()

print(f"DataLoaders loaded in {t1 - t0:.2f} seconds!")
print(f"Train batches: {len(train_l)} (N = {len(train_l.dataset):,})")
print(f"Val batches:   {len(val_l)} (N = {len(val_l.dataset):,})")
print(f"Test batches:  {len(test_l)} (N = {len(test_l.dataset):,})")
print(f"Positive weights for BCE Loss: {pos_w.tolist()}")

# Fetch one batch
x, y, ecg_id = next(iter(train_l))
print(f"Sample Batch - x: {x.shape} (dtype={x.dtype}), y: {y.shape} (dtype={y.dtype})")
print("Data validation successful!")
