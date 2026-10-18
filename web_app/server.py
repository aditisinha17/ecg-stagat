"""
ECG-STGAT Interactive Web Application Server.
Provides REST API and serves the frontend dashboard on localhost.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np
import pandas as pd
import torch

from ecg_stgat.data.ptbxl_dataset import DIAGNOSTIC_CLASSES, LEAD_NAMES, load_raw_ptbxl_metadata
from ecg_stgat.models.stgat_model import STGATModel
from ecg_stgat.models.baseline_cnn import BaselineCNNModel
from ecg_stgat.explainability.saliency import compute_smoothgrad
from ecg_stgat.explainability.waveform_alignment import (
    segment_ecg_intervals,
    compute_interval_alignment,
    compute_attribution_faithfulness,
)

PORT = 8000
DATA_DIR = Path("data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

# Global Cache for Models & Data
CACHED_MODELS = {}
CACHED_SIGNALS = None
CACHED_METADATA = None
CACHED_TEST_IDS = []


def init_backend():
    global CACHED_MODELS, CACHED_SIGNALS, CACHED_METADATA, CACHED_TEST_IDS
    print("[Server Init] Preloading models and test dataset...")

    device = torch.device("cpu")

    # 1. Load ST-GAT Static Model
    stgat_p = Path("runs/STGAT_Static/best_model.pt")
    if stgat_p.exists():
        ckpt = torch.load(stgat_p, map_location=device)
        model = STGATModel(embed_dim=128, num_classes=5, num_heads=4, num_gat_layers=2, graph_type="static").to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        CACHED_MODELS["stgat"] = model
        print("  Loaded ST-GAT (Static Graph)")

    # 2. Load Baseline CNN Model
    base_p = Path("runs/Baseline_CNN/best_model.pt")
    if base_p.exists():
        ckpt = torch.load(base_p, map_location=device)
        model = BaselineCNNModel(embed_dim=128, num_classes=5).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        CACHED_MODELS["baseline"] = model
        print("  Loaded Baseline 1D-CNN")

    # 3. Load ST-GAT Lead-Dropout Model
    drop_p = Path("runs/STGAT_LeadDropout/best_model.pt")
    if drop_p.exists():
        ckpt = torch.load(drop_p, map_location=device)
        model = STGATModel(embed_dim=128, num_classes=5, num_heads=4, num_gat_layers=2, graph_type="static").to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        CACHED_MODELS["dropout"] = model
        print("  Loaded ST-GAT (Lead-Dropout Regularized)")

    # 3. Load Fold 10 Test Signals
    test_sig_p = Path("data/cache/signals_folds_10.npy")
    if test_sig_p.exists():
        CACHED_SIGNALS = np.load(test_sig_p, mmap_mode="r")
        print(f"  Loaded Memory-Mapped Test Signals: {CACHED_SIGNALS.shape}")

    # 4. Load Metadata for Fold 10
    df, diag_scp = load_raw_ptbxl_metadata(str(DATA_DIR))
    test_df = df[df["strat_fold"] == 10].copy().reset_index()

    from ecg_stgat.data.ptbxl_dataset import map_scp_to_superclasses
    test_df["diagnostic_superclass"] = test_df["scp_codes_dict"].apply(
        lambda scps: map_scp_to_superclasses(scps, diag_scp)
    )
    for c in DIAGNOSTIC_CLASSES:
        test_df[c] = test_df["diagnostic_superclass"].apply(lambda clss: int(c in clss))

    CACHED_METADATA = test_df
    CACHED_TEST_IDS = test_df["ecg_id"].tolist()
    print(f"  Loaded {len(test_df)} Test Fold 10 records.")


class ECGRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self.serve_file(Path("web_app/index.html"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.send_json({
                "status": "online",
                "models_loaded": list(CACHED_MODELS.keys()),
                "total_test_records": len(CACHED_TEST_IDS),
                "diagnostic_classes": DIAGNOSTIC_CLASSES,
                "lead_names": LEAD_NAMES,
            })
        elif path == "/api/records":
            self.get_records_list(query)
        elif path == "/api/waveform":
            self.get_waveform(query)
        elif path == "/api/graph_topology":
            self.get_graph_topology()
        elif path == "/api/tables":
            self.get_tables_data()
        elif path.startswith("/figures/"):
            fig_name = path.replace("/figures/", "")
            fig_path = Path("results/figures") / fig_name
            if fig_path.exists():
                ext = fig_path.suffix.lower()
                mime = "image/svg+xml" if ext == ".svg" else "image/png"
                self.serve_file(fig_path, mime)
            else:
                self.send_error(404, "Figure not found")
        else:
            # Check web_app static files
            static_p = Path("web_app") / path.lstrip("/")
            if static_p.exists() and static_p.is_file():
                ext = static_p.suffix.lower()
                mime = "text/css" if ext == ".css" else ("application/javascript" if ext == ".js" else "text/plain")
                self.serve_file(static_p, mime)
            else:
                self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        data = json.loads(body) if body else {}

        if path == "/api/predict":
            self.handle_predict(data)
        elif path == "/api/explain":
            self.handle_explain(data)
        else:
            self.send_error(404, "Not Found")

    def send_json(self, data: Any, code: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, file_path: Path, mime_type: str):
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def get_records_list(self, query):
        records = []
        limit = int(query.get("limit", [30])[0])
        superclass_filter = query.get("class", [None])[0]

        df = CACHED_METADATA
        if superclass_filter and superclass_filter in DIAGNOSTIC_CLASSES:
            df = df[df[superclass_filter] == 1]

        subset = df.head(limit)
        for _, row in subset.iterrows():
            active_super = [c for c in DIAGNOSTIC_CLASSES if row[c] == 1]
            records.append({
                "index": int(row.name),
                "ecg_id": int(row["ecg_id"]),
                "age": int(row["age"]) if pd.notna(row["age"]) else None,
                "sex": "Male" if row["sex"] == 0 else "Female",
                "superclasses": active_super,
                "patient_id": int(row["patient_id"]) if pd.notna(row["patient_id"]) else None,
            })
        self.send_json(records)

    def get_waveform(self, query):
        idx = int(query.get("index", [0])[0])
        if CACHED_SIGNALS is None or idx < 0 or idx >= len(CACHED_SIGNALS):
            self.send_error(400, "Invalid index")
            return

        sig = np.array(CACHED_SIGNALS[idx], dtype=float)  # (12, 1000)
        meta_row = CACHED_METADATA.iloc[idx]
        active_super = [c for c in DIAGNOSTIC_CLASSES if meta_row[c] == 1]

        # Extract wave interval segmentation on Lead II
        lead_ii = sig[1]
        seg_res = segment_ecg_intervals(lead_ii, fs=100.0)

        self.send_json({
            "index": idx,
            "ecg_id": int(meta_row["ecg_id"]),
            "age": int(meta_row["age"]) if pd.notna(meta_row["age"]) else None,
            "sex": "Male" if meta_row["sex"] == 0 else "Female",
            "superclasses": active_super,
            "signals": sig.tolist(),
            "lead_names": LEAD_NAMES,
            "sampling_rate": 100,
            "duration_sec": 10.0,
            "intervals": {
                "delineation_method": seg_res["delineation_method"],
                "num_beats": seg_res["num_delineated_beats"],
                "qrs_mask": seg_res["qrs"].tolist(),
                "st_segment_mask": seg_res["st_segment"].tolist(),
                "t_wave_mask": seg_res["t_wave"].tolist(),
                "p_wave_mask": seg_res["p_wave"].tolist(),
            }
        })

    def get_graph_topology(self):
        from ecg_stgat.models.lead_graph import build_static_anatomical_adjacency
        adj = build_static_anatomical_adjacency().cpu().numpy()
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                if adj[i, j] > 0.0:
                    edges.append({
                        "source": LEAD_NAMES[i],
                        "target": LEAD_NAMES[j],
                        "source_idx": i,
                        "target_idx": j,
                    })
        self.send_json({
            "lead_names": LEAD_NAMES,
            "adjacency_matrix": adj.tolist(),
            "edges": edges,
            "num_edges": len(edges),
            "self_loops": LEAD_NAMES,
            "num_self_loops": 12,
            "graph_type": "static_anatomical",
            "description": "PTB-XL 12-lead static anatomical adjacency defined in ecg_stgat.models.lead_graph.build_static_anatomical_adjacency"
        })

    def handle_predict(self, data):
        idx = int(data.get("index", 0))
        model_name = data.get("model", "stgat").lower()
        active_leads = data.get("active_leads", list(range(12)))

        if model_name not in CACHED_MODELS:
            model_name = "stgat"
        model = CACHED_MODELS[model_name]

        sig = np.array(CACHED_SIGNALS[idx], dtype=np.float32)  # (12, 1000)
        mask = np.zeros_like(sig)
        mask[active_leads, :] = 1.0
        masked_sig = sig * mask

        x_t = torch.from_numpy(masked_sig).unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            if model_name in ("stgat", "dropout"):
                logits, exp = model(x_t, return_explainability=True)
                lead_weights = exp["lead_pooling_weights"].squeeze(0).detach().cpu().numpy().tolist()
                gat_att = exp["gat_attention"].squeeze(0).mean(dim=0).detach().cpu().numpy().tolist()
                gat_att_heads = exp["gat_attention"].squeeze(0).detach().cpu().numpy().tolist()
            else:
                logits, exp = model(x_t, return_explainability=True)
                lead_weights = exp["lead_pooling_weights"].squeeze(0).detach().cpu().numpy().tolist()
                gat_att = None
                gat_att_heads = None

            probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().tolist()
        elapsed_ms = (time.time() - t0) * 1000.0

        meta_row = CACHED_METADATA.iloc[idx]
        true_super = [c for c in DIAGNOSTIC_CLASSES if meta_row[c] == 1]

        self.send_json({
            "model": model_name,
            "active_leads": active_leads,
            "probabilities": {DIAGNOSTIC_CLASSES[i]: probs[i] for i in range(5)},
            "binary_predictions": {DIAGNOSTIC_CLASSES[i]: bool(probs[i] >= 0.5) for i in range(5)},
            "ground_truth": true_super,
            "lead_weights": lead_weights,
            "gat_attention_matrix": gat_att,
            "gat_attention_heads": gat_att_heads,
            "inference_time_ms": elapsed_ms,
        })

    def handle_explain(self, data):
        idx = int(data.get("index", 0))
        target_class = data.get("target_class", "NORM").upper()
        if target_class not in DIAGNOSTIC_CLASSES:
            target_class = "NORM"
        c_idx = DIAGNOSTIC_CLASSES.index(target_class)

        model = CACHED_MODELS.get("stgat")
        if model is None:
            model = list(CACHED_MODELS.values())[0]

        sig = np.array(CACHED_SIGNALS[idx], dtype=np.float32)
        x_t = torch.from_numpy(sig).unsqueeze(0)

        # Compute SmoothGrad saliency
        sal = compute_smoothgrad(model, x_t, target_class_idx=c_idx, num_samples=10)
        align = compute_interval_alignment(sal, sig, fs=100.0)
        faith = compute_attribution_faithfulness(model, x_t, sal, target_class_idx=c_idx)

        # Average attribution per lead
        lead_attribution_mass = sal.sum(axis=1).tolist()

        self.send_json({
            "target_class": target_class,
            "target_class_idx": c_idx,
            "saliency_lead_ii": sal[1].tolist(),
            "saliency_mean": sal.mean(axis=0).tolist(),
            "lead_attribution_mass": lead_attribution_mass,
            "interval_percentages": {
                "QRS Complex": align["qrs_overlap_pct"],
                "ST Segment": align["st_segment_overlap_pct"],
                "T-Wave": align["t_wave_overlap_pct"],
                "ST-T Combined": align["st_t_overlap_pct"],
                "P-Wave": align["p_wave_overlap_pct"],
                "Isoelectric Baseline": align["baseline_overlap_pct"],
            },
            "delineation_meta": {
                "method": align["delineation_method"],
                "valid_beats": align["num_delineated_beats"],
                "success": align["delineation_success"],
            },
            "faithfulness": {
                "faithfulness_ratio": faith["faithfulness_ratio"],
                "salient_ablation_drop": faith["salient_drop"],
                "random_ablation_drop": faith["random_drop"],
            }
        })

    def get_tables_data(self):
        tables = {}
        for t_name in ["table1_rq1_model_performance", "table2_per_class_breakdown", "table3_rq2_ablation_study", "table4_rq4_missing_lead_robustness", "table5_rq3_interval_explainability"]:
            p = Path("results/tables") / f"{t_name}.csv"
            if p.exists():
                df = pd.read_csv(p)
                tables[t_name] = df.to_dict(orient="records")
        self.send_json(tables)


def run_server(port: int = PORT):
    init_backend()
    server_address = ("", port)
    httpd = HTTPServer(server_address, ECGRequestHandler)
    print(f"\n=======================================================")
    print(f"[*] ECG-STGAT Localhost Dashboard Running Live!")
    print(f"    URL: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
