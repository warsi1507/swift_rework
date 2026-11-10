"""
ml/src/abc_driver.py
Subprocess Driver for ABC Mapping, Genlib Deduplication, and QoR Extraction.
"""

import os
import re
import time
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


@dataclass
class QoRMetrics:
    area: float = 0.0
    delay: float = 0.0
    adp: float = 0.0
    num_gates: int = 0
    num_levels: int = 0
    map_time: float = 0.0
    total_time: float = 0.0
    success: bool = False
    error: str = ""

    def to_dict(self):
        return asdict(self)


def deduplicate_genlib_gate_names(src_genlib: Path, out_genlib: Path) -> Tuple[Path, int]:
    """Remove repeated gate-name blocks in a .genlib file."""
    src_genlib = src_genlib.resolve()
    out_genlib = out_genlib.resolve()
    out_genlib.parent.mkdir(parents=True, exist_ok=True)

    lines = src_genlib.read_text().splitlines()
    header: List[str] = []
    blocks: List[List[str]] = []
    current_block: List[str] = []

    for line in lines:
        if line.lstrip().startswith("GATE "):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            if current_block:
                current_block.append(line)
            else:
                header.append(line)

    if current_block:
        blocks.append(current_block)

    kept_blocks: List[List[str]] = []
    seen_names = set()
    dropped = 0

    for block in blocks:
        first = block[0].strip()
        match = re.match(r"^GATE\s+(\S+)", first)
        if not match:
            kept_blocks.append(block)
            continue

        gate_name = match.group(1)
        if gate_name in seen_names:
            dropped += 1
            continue

        seen_names.add(gate_name)
        kept_blocks.append(block)

    out_lines: List[str] = []
    out_lines.extend(header)
    for block in kept_blocks:
        out_lines.extend(block)

    out_genlib.write_text("\n".join(out_lines) + "\n")
    return out_genlib, dropped


class ABCDriver:
    def __init__(self, abc_bin: str = "abc/abc"):
        self.abc_bin = Path(abc_bin).resolve()
        if not self.abc_bin.exists():
            raise FileNotFoundError(f"ABC executable not found at: {self.abc_bin}")

    def run_mapping(self, 
                    circuit_aig: str, 
                    genlib_path: str, 
                    super_lib_path: str = None, 
                    timeout: int = 300) -> QoRMetrics:
        metrics = QoRMetrics()
        circuit_abs = Path(circuit_aig).resolve()
        genlib_abs = Path(genlib_path).resolve()
        
        script_commands = [
            f"read_aiger {circuit_abs}",
            "strash",
            f"read_library {genlib_abs}"
        ]
        
        if super_lib_path:
            super_abs = Path(super_lib_path).resolve()
            script_commands.append(f"read_super {super_abs}")
            
        script_commands.extend([
            "echo __MAP_TIME_START__",
            "time",
            "map",
            "time",
            "echo __MAP_TIME_END__",
            "print_stats",
            "quit"
        ])
        
        script_content = "\n".join(script_commands)
        start_wall = time.perf_counter()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.abc', delete=False) as f:
            f.write(script_content)
            tmp_path = f.name
            
        try:
            res = subprocess.run(
                [str(self.abc_bin), "-f", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            metrics.total_time = time.perf_counter() - start_wall
            
            if res.returncode != 0:
                metrics.error = res.stderr
                return metrics
                
            stdout = res.stdout
            
            # Extract internal mapping elapsed time (the last "time" report
            # between the markers belongs to the map command itself).
            # The last marker pair is used because ABC echoes earlier
            # sessions' commands from abc.history at startup.
            timing_segments = re.findall(r'__MAP_TIME_START__(.*?)__MAP_TIME_END__', stdout, re.DOTALL)
            if timing_segments:
                elapsed = re.findall(r'elapse:\s*([\d.]+)\s*seconds', timing_segments[-1])
                metrics.map_time = float(elapsed[-1]) if elapsed else metrics.total_time
            else:
                metrics.map_time = metrics.total_time
                
            # Primary regex parse: nd = X, area = Y, delay = Z, lev = W
            stat_match = re.search(r'nd\s*=\s*(\d+).*?area\s*=\s*([\d.]+).*?delay\s*=\s*([\d.]+).*?lev\s*=\s*(\d+)', 
                                   stdout, flags=re.IGNORECASE | re.DOTALL)
            if stat_match:
                metrics.num_gates = int(stat_match.group(1))
                metrics.area = float(stat_match.group(2))
                metrics.delay = float(stat_match.group(3))
                metrics.num_levels = int(stat_match.group(4))
                metrics.adp = metrics.area * metrics.delay
                metrics.success = True
            else:
                # Fallback parse for decoupled area/delay reporting
                area_m = re.findall(r"Area\s*=\s*([\d.]+)", stdout, flags=re.IGNORECASE)
                delay_m = re.findall(r"Delay\s*=\s*([\d.]+)", stdout, flags=re.IGNORECASE)
                if area_m and delay_m:
                    metrics.area = float(area_m[-1])
                    metrics.delay = float(delay_m[-1])
                    metrics.adp = metrics.area * metrics.delay
                    metrics.success = True
                else:
                    metrics.error = "Could not parse ABC mapping stats."
                
        except subprocess.TimeoutExpired:
            metrics.error = f"ABC mapping timed out after {timeout} seconds."
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return metrics