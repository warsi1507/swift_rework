"""
ml/src/dataset.py
Binary AIGER parsing for AIG graph features, dataset construction with
strict train/val/test partitioning, and PyG batch collation.
"""

import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


UNSEEN_CIRCUITS = [
    'hyp', 'div_alu2', 'div_apex7', 'div',
    'mem_ctrl_alu2', 'mem_ctrl_apex7', 'mem_ctrl',
    'mult_wallace_orig', 'log2_alu2', 'log2_apex7',
    'mult_dadda_orig', 'log2', 'multiplier_alu2',
    'multiplier_apex7', 'multiplier'
]


def _read_unsigned(data: bytes, pos: int):
    """Reads one AIGER delta-encoded unsigned integer, returning (value, new_pos)."""
    x = 0
    i = 0
    while True:
        ch = data[pos]
        pos += 1
        x |= (ch & 0x7f) << (7 * i)
        if not (ch & 0x80):
            return x, pos
        i += 1


def parse_aiger(path: str) -> dict:
    """Parses a binary AIGER file into node types, inverted-fanin counts, and edges."""
    raw = Path(path).read_bytes()
    newline = raw.index(b"\n")
    header = raw[:newline].decode("ascii").split()
    if len(header) != 6 or header[0] != "aig":
        raise ValueError(f"Not a binary AIGER header in {path}: {header}")
    n_max, n_ins, n_lats, n_outs, n_ands = map(int, header[1:])
    if n_lats != 0:
        raise ValueError(f"AIGER file {path} contains latches; not supported")
    if n_max != n_ins + n_lats + n_ands:
        raise ValueError(f"AIGER header mismatch in {path}: M={n_max}, I+L+A={n_ins + n_lats + n_ands}")

    pos = newline + 1
    num_nodes = 1 + n_max + n_outs  # const node 0 + max var + PO nodes

    node_type = np.zeros(num_nodes, dtype=np.int64)       # 0: const/PI
    num_inverted = np.zeros(num_nodes, dtype=np.int64)
    edge_src, edge_dst = [], []

    # AND nodes (type 2)
    node_type[1 + n_ins: 1 + n_ins + n_ands] = 2

    # PO nodes (type 1): one ASCII literal each
    for po in range(n_outs):
        line_end = raw.index(b"\n", pos)
        lit = int(raw[pos:line_end].decode("ascii").strip())
        node = 1 + n_max + po
        node_type[node] = 1
        edge_src.append(lit >> 1)
        edge_dst.append(node)
        if lit & 1:
            num_inverted[node] += 1
        pos = line_end + 1

    # AND gates: two delta-encoded fanin literals each
    for i in range(n_ands):
        node_id = 1 + n_ins + i
        u_lit = 2 * node_id
        d1, pos = _read_unsigned(raw, pos)
        lit1 = u_lit - d1
        d0, pos = _read_unsigned(raw, pos)
        lit0 = lit1 - d0
        for lit in (lit0, lit1):
            edge_src.append(lit >> 1)
            edge_dst.append(node_id)
            if lit & 1:
                num_inverted[node_id] += 1

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64) if edge_src else np.zeros((2, 0), dtype=np.int64)
    return {
        'node_type': node_type,
        'num_inverted_predecessors': num_inverted,
        'edge_index': edge_index,
        'num_nodes': num_nodes
    }


def aig_graph_to_pyg(graph_dict: dict):
    return Data(
        node_type=torch.tensor(graph_dict['node_type'], dtype=torch.long),
        num_inverted_predecessors=torch.tensor(graph_dict['num_inverted_predecessors'], dtype=torch.long),
        edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
        num_nodes=graph_dict['num_nodes']
    )


def parse_truth_table_32bit(tt_hex: str) -> np.ndarray:
    try:
        s = tt_hex.strip()
        if s.lower().startswith('0x'):
            s = s[2:]
        val = int(s, 16)
        binary_32 = format(val, '032b')
        return np.array([float(b) for b in binary_32[:32]], dtype=np.float32)
    except Exception:
        return np.zeros(32, dtype=np.float32)


class SupergateDataset(Dataset):
    def __init__(self, 
                 csv_path: str, 
                 split: str = 'train', 
                 train_val_ratio: float = 0.8,
                 num_samples: int = None, 
                 seed: int = 42):
        self.csv_path = csv_path
        self.split = split
        
        all_rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append(r)
                
        circuit_names = sorted(list(set(Path(r['AIG_Path']).stem for r in all_rows)))
        test_set = set(UNSEEN_CIRCUITS)
        remaining_circuits = [c for c in circuit_names if c not in test_set]
        
        rng = random.Random(seed)
        rng.shuffle(remaining_circuits)
        
        num_train = int(len(remaining_circuits) * train_val_ratio)
        train_set = set(remaining_circuits[:num_train])
        val_set = set(remaining_circuits[num_train:])
        
        if split == 'test':
            selected_circuits = test_set
        elif split == 'val':
            selected_circuits = val_set
        else:
            selected_circuits = train_set
            
        self.data_rows = [r for r in all_rows if Path(r['AIG_Path']).stem in selected_circuits]
        
        if num_samples and num_samples < len(self.data_rows):
            rng.shuffle(self.data_rows)
            self.data_rows = self.data_rows[:num_samples]
            
        unique_paths = list(set(r['AIG_Path'] for r in self.data_rows))
        self.graph_cache = {}
        for path in unique_paths:
            if os.path.exists(path):
                self.graph_cache[path] = aig_graph_to_pyg(parse_aiger(path))
            else:
                print(f"Warning: Missing circuit AIG: {path}")
                
        areas = [float(r['Area']) for r in self.data_rows]
        delays = [float(r['Delay_Max']) for r in self.data_rows]
        self.area_min, self.area_max = (min(areas), max(areas)) if areas else (0.0, 1.0)
        self.delay_min, self.delay_max = (min(delays), max(delays)) if delays else (0.0, 1.0)

    def __len__(self) -> int:
        return len(self.data_rows)

    def __getitem__(self, idx: int):
        row = self.data_rows[idx]
        graph = self.graph_cache.get(row['AIG_Path'])
        
        area = float(row['Area'])
        delay = float(row['Delay_Max'])
        a_norm = (area - self.area_min) / (self.area_max - self.area_min + 1e-8)
        d_norm = (delay - self.delay_min) / (self.delay_max - self.delay_min + 1e-8)
        
        sg_data = {
            'truth_table': torch.tensor(parse_truth_table_32bit(row['Truth_Table_Hex'])),
            'area': torch.tensor(a_norm, dtype=torch.float32),
            'delay': torch.tensor(d_norm, dtype=torch.float32),
            'num_inputs': torch.tensor(int(row['Num_Inputs']), dtype=torch.long)
        }
        label = torch.tensor(float(row['Label']), dtype=torch.float32)
        return graph, sg_data, label


def collate_batch(batch):
    graphs = [item[0] for item in batch]
    sg_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    batched_graphs = Batch.from_data_list(graphs)
    batched_sg = {
        'truth_table': torch.stack([s['truth_table'] for s in sg_list]),
        'area': torch.stack([s['area'] for s in sg_list]),
        'delay': torch.stack([s['delay'] for s in sg_list]),
        'num_inputs': torch.stack([s['num_inputs'] for s in sg_list]),
    }
    return batched_graphs, batched_sg, torch.stack(labels)
