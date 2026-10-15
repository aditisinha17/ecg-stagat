"""
Smoke train verification on small subset.
"""
import yaml
from ecg_stgat.training.train import train_experiment

with open("ecg_stgat/training/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Quick smoke test settings
cfg["training"]["epochs"] = 2
cfg["data"]["train_folds"] = [1]  # 1 fold only
cfg["data"]["batch_size"] = 32

print("\n--- Smoke Test 1: Baseline 1D-CNN ---")
cfg["model"]["name"] = "baseline_cnn"
train_experiment(config=cfg, exp_name="smoke_baseline")

print("\n--- Smoke Test 2: ST-GAT ---")
cfg["model"]["name"] = "stgat"
cfg["model"]["graph_type"] = "static"
train_experiment(config=cfg, exp_name="smoke_stgat")

print("\n>>> SMOKE TRAIN COMPLETED SUCCESSFULLY! <<<")
