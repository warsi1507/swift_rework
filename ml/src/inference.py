"""
ml/src/inference.py
Two-stage supergate selection: a global frequency core plus
model-scored candidates, with top-up filling and checkpoint discovery.
"""

import os
import json
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from models import GNNCompatibilityModel
from dataset import parse_aiger, aig_graph_to_pyg, parse_truth_table_32bit
from supergate_lib import SupergateLibraryWriter


def discover_best_checkpoint(models_root: str) -> Path:
    root = Path(models_root).resolve()
    if root.is_file() and root.suffix == ".pth":
        return root

    best_ckpt, best_f1 = None, float("-inf")
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt = run_dir / "best_model.pth"
        hist = run_dir / "training_history.json"
        if not ckpt.exists():
            continue

        if hist.exists():
            try:
                data = json.loads(hist.read_text())
                if isinstance(data, list):
                    f1s = [float(x.get("f1", -1)) for x in data if isinstance(x, dict)]
                    max_f1 = max(f1s) if f1s else -1
                elif isinstance(data, dict):
                    max_f1 = float(data.get("f1", -1))
                else:
                    max_f1 = -1
                if max_f1 > best_f1:
                    best_f1 = max_f1
                    best_ckpt = ckpt
            except Exception:
                pass
        elif best_ckpt is None:
            best_ckpt = ckpt

    if best_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in: {models_root}")
    return best_ckpt


class SupergateSelector:
    def __init__(self,
                 checkpoint_path: Optional[str],
                 supergates_csv: str,
                 global_freq_csv: str,
                 source_super_file: str,
                 genlib_path: str = "data/library/library.genlib",
                 device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not checkpoint_path or not Path(checkpoint_path).exists():
            ckpt_file = discover_best_checkpoint("ml/checkpoints")
        else:
            ckpt_file = Path(checkpoint_path).resolve()

        print(f"Loading GNN compatibility model from: {ckpt_file}")
        self.model = GNNCompatibilityModel().to(self.device)
        ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        self.model.eval()

        self.supergates_df = pd.read_csv(supergates_csv)
        self.freq_df = pd.read_csv(global_freq_csv)

        tt_to_id = {
            str(tt).strip().lower(): int(sid)
            for tt, sid in zip(self.supergates_df['truth_table_hex'], self.supergates_df['supergate_id'])
        }

        if 'total_usage' in self.freq_df.columns:
            sorted_core = self.freq_df.sort_values(by='total_usage', ascending=False)
        else:
            usage_cols = [c for c in self.freq_df.columns if c not in ('circuit', 'AIG_Path')]
            tt_cols = [c for c in usage_cols if c.strip().lower() in tt_to_id]
            missing = [c for c in usage_cols if c.strip().lower() not in tt_to_id]
            if missing:
                print(f"Warning: {len(missing)} frequency columns missing from truth-table CSV; skipped")
            total = self.freq_df[tt_cols].sum(axis=0)
            sorted_core = pd.DataFrame({
                'supergate_id': [tt_to_id[c.strip().lower()] for c in tt_cols],
                'total_usage': total.values
            }).sort_values(by='total_usage', ascending=False)

        self.global_rankings = sorted_core['supergate_id'].tolist()
        self.lib_writer = SupergateLibraryWriter(source_super_file, genlib_path)
        self._precompute_supergate_tensors()

    def _precompute_supergate_tensors(self):
        areas = self.supergates_df['area'].values
        delays = self.supergates_df['delay'].values
        a_min, a_max = areas.min(), areas.max()
        d_min, d_max = delays.min(), delays.max()

        tts = [parse_truth_table_32bit(x) for x in self.supergates_df['truth_table_hex']]
        areas_norm = (areas - a_min) / (a_max - a_min + 1e-8)
        delays_norm = (delays - d_min) / (d_max - d_min + 1e-8)
        inputs = self.supergates_df['num_inputs'].values

        self.tensor_tts = torch.tensor(np.array(tts), dtype=torch.float32)
        self.tensor_areas = torch.tensor(areas_norm, dtype=torch.float32)
        self.tensor_delays = torch.tensor(delays_norm, dtype=torch.float32)
        self.tensor_inputs = torch.tensor(inputs, dtype=torch.long)
        self.all_ids = self.supergates_df['supergate_id'].tolist()

    def select_supergates(self, circuit_aig: str, k: int = 500, m_core: int = 100) -> Tuple[List[int], float]:
        start_time = time.perf_counter()

        # Stage 1: Global Core Selection (M = 100)
        core_gates = self.global_rankings[:m_core]
        selected_set = set(core_gates)
        selected_list = list(core_gates)

        needed_ml = k - len(selected_list)
        if needed_ml <= 0:
            return selected_list[:k], time.perf_counter() - start_time

        # Stage 2: ML-Guided Scoring
        graph_dict = parse_aiger(circuit_aig)
        pyg_data = aig_graph_to_pyg(graph_dict).to(self.device)

        with torch.no_grad():
            circuit_emb = self.model.encode_circuit(pyg_data)
            non_core_indices = [idx for idx, sg_id in enumerate(self.all_ids) if sg_id not in selected_set]

            sg_data_batch = {
                'truth_table': self.tensor_tts[non_core_indices].to(self.device),
                'area': self.tensor_areas[non_core_indices].to(self.device),
                'delay': self.tensor_delays[non_core_indices].to(self.device),
                'num_inputs': self.tensor_inputs[non_core_indices].to(self.device)
            }
            scores = self.model.score_from_embedding(circuit_emb, sg_data_batch).cpu().numpy()

        top_indices = np.argsort(-scores)[:needed_ml]
        for idx in top_indices:
            sg_id = self.all_ids[non_core_indices[idx]]
            if sg_id not in selected_set:
                selected_set.add(sg_id)
                selected_list.append(sg_id)

        # Top-up fallback: ensure library meets target K gates
        if len(selected_list) < k:
            for fallback_id in self.global_rankings:
                if fallback_id not in selected_set:
                    selected_set.add(fallback_id)
                    selected_list.append(fallback_id)
                if len(selected_list) >= k:
                    break

        inference_time = time.perf_counter() - start_time
        return selected_list[:k], inference_time

    def generate_library_file(self, selected_ids: List[int], output_path: str) -> Path:
        return self.lib_writer.write_reduced_library(selected_ids, output_path)