#!/usr/bin/env python3
"""Baseline QoR analysis with the ASAP7 7nm library.

Maps benchmark circuits with the standard unaugmented 7nm.lib (no
supergates) and records baseline area, delay, ADP, node count, logic
levels, and runtime.
"""

import csv
import re
import subprocess
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, asdict

# ==========================================
# Directory Configuration
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SRC_DIR.parent
PROJECT_ROOT = ANALYSIS_DIR.parent

ABC_BIN = PROJECT_ROOT / "abc" / "abc"
CIRCUITS_DIR = PROJECT_ROOT / "data" / "benchmarks"
DEFAULT_LIB = PROJECT_ROOT / "data" / "library" / "7nm.lib"

LOG_DIR = ANALYSIS_DIR / "results" / "logs" / "qor_baseline"
CSV_OUT = ANALYSIS_DIR / "results" / "asap7_baseline_qor.csv"
LOG_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class QorResult:
    circuit: str
    area: float = None
    delay: float = None
    adp: float = None
    levels: int = None
    nodes: int = None
    edges: int = None
    runtime_s: float = None
    success: bool = False
    error: str = ""

    def to_dict(self):
        return asdict(self)

def parse_abc_metrics(output: str) -> dict:
    metrics = {"area": None, "delay": None, "levels": None, "nodes": None, "edges": None}
    
    mapped_line = None
    for line in reversed(output.splitlines()):
        if "area" in line.lower() and "delay" in line.lower():
            mapped_line = line.strip()
            break

    if mapped_line:
        m = re.search(r"nd\s*=\s*(\d+).*?edge\s*=\s*(\d+).*?area\s*=\s*([0-9]+(?:\.[0-9]+)?).*?delay\s*=\s*([0-9]+(?:\.[0-9]+)?).*?lev\s*=\s*(\d+)", mapped_line, re.IGNORECASE)
        if m:
            metrics.update({"nodes": int(m.group(1)), "edges": int(m.group(2)), "area": float(m.group(3)), "delay": float(m.group(4)), "levels": int(m.group(5))})
            return metrics

    # Fallback to loose regex mapping
    for key, pattern in [("area", r"area\s*=\s*([0-9]+(?:\.[0-9]+)?)"), ("delay", r"delay\s*=\s*([0-9]+(?:\.[0-9]+)?)"), 
                         ("levels", r"lev\s*=\s*(\d+)"), ("nodes", r"nd\s*=\s*(\d+)"), ("edges", r"edge\s*=\s*(\d+)")]:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            metrics[key] = float(matches[-1]) if key in ["area", "delay"] else int(matches[-1])
            
    return metrics

def run_mapping(circuit_path: Path) -> QorResult:
    script_text = f"read {circuit_path.absolute()}\nread_lib {DEFAULT_LIB.absolute()}\nmap -v\nprint_stats\nquit\n"
    start = time.perf_counter()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".abc", delete=False) as temp_script:
        temp_script.write(script_text)
        temp_script_path = Path(temp_script.name)

    log_path = LOG_DIR / f"{circuit_path.stem}.log"
    try:
        completed = subprocess.run([str(ABC_BIN), "-f", str(temp_script_path)], capture_output=True, text=True, timeout=1800)
        runtime_s = time.perf_counter() - start
        combined_output = completed.stdout + "\n" + completed.stderr
        log_path.write_text(combined_output)

        metrics = parse_abc_metrics(combined_output)
        area, delay = metrics["area"], metrics["delay"]
        success = completed.returncode == 0 and area is not None and delay is not None
        
        return QorResult(
            circuit=circuit_path.stem, area=area, delay=delay, adp=(area * delay) if success else None,
            levels=metrics["levels"], nodes=metrics["nodes"], edges=metrics["edges"],
            runtime_s=runtime_s, success=success, error="" if success else f"Exit code {completed.returncode}"
        )
    except subprocess.TimeoutExpired:
        return QorResult(circuit=circuit_path.stem, success=False, error="Timeout")
    except Exception as exc:
        return QorResult(circuit=circuit_path.stem, success=False, error=str(exc))
    finally:
        if temp_script_path.exists():
            temp_script_path.unlink()

def main():
    if not ABC_BIN.exists() or not DEFAULT_LIB.exists():
        print("ERROR: Ensure ABC is compiled and 7nm.lib exists in data/library/")
        return

    circuits = list(CIRCUITS_DIR.glob("*.aig"))
    results = [run_mapping(circuit) for circuit in circuits]

    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(QorResult("").to_dict().keys()))
        writer.writeheader()
        writer.writerows([r.to_dict() for r in results])

    successes = [r for r in results if r.success]
    print(f"\nCompleted {len(successes)}/{len(circuits)} mappings.")
    print(f"Results saved to {CSV_OUT}")

if __name__ == "__main__":
    main()