"""
ml/src/supergate_lib.py
DAG-Safe Supergate Library Slicing, Dependency Closure, and Remapping.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def patch_genlib_line(src_super: Path, dst_super: Path, genlib_abs: Path) -> Path:
    """Copies a .super file with its genlib reference line replaced."""
    src_super = Path(src_super).resolve()
    dst_super = Path(dst_super).resolve()
    genlib_abs = Path(genlib_abs).resolve()
    dst_super.parent.mkdir(parents=True, exist_ok=True)

    lines = src_super.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, s in enumerate(lines):
        t = s.strip()
        if t and not t.startswith("#"):
            lines[i] = str(genlib_abs)
            break
    else:
        raise ValueError(f"No genlib reference line found in {src_super}")

    dst_super.write_text("\n".join(lines) + "\n")
    return dst_super


@dataclass
class SuperLine:
    old_body_id: int
    gate_name: str
    fanins: List[int]
    is_real_super: bool


class SupergateLibraryWriter:
    def __init__(self, source_super_file: str, genlib_path: str = "data/library/library.genlib"):
        self.source_super_file = Path(source_super_file).resolve()
        self.genlib_path = Path(genlib_path).resolve()
        
        if not self.source_super_file.exists():
            raise FileNotFoundError(f"Source super library not found: {self.source_super_file}")
            
        self._parse()

    def _parse(self):
        lines = self.source_super_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        non_comment = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            non_comment.append((idx, stripped))

        if len(non_comment) < 4:
            raise ValueError(f"Invalid .super header in {self.source_super_file}")

        self.genlib_token = non_comment[0][1]
        self.n_vars = int(non_comment[1][1])
        self.declared_n_supers = int(non_comment[2][1])
        self.declared_n_lines = int(non_comment[3][1])

        body_start_line = non_comment[3][0] + 1
        self.body_lines: List[SuperLine] = []

        for line in lines[body_start_line:]:
            core = line.split("#", 1)[0].strip()
            if not core:
                continue

            parts = core.split()
            is_real = (parts[0] == "*")
            gate_idx = 1 if is_real else 0
            if gate_idx >= len(parts):
                continue

            gate_name = parts[gate_idx]
            fanins = [int(tok) for tok in parts[gate_idx + 1:]]
            body_id = len(self.body_lines)
            
            self.body_lines.append(
                SuperLine(
                    old_body_id=body_id,
                    gate_name=gate_name,
                    fanins=fanins,
                    is_real_super=is_real,
                )
            )

        self._validate_basic()

    def _validate_basic(self):
        max_abs_index = self.n_vars + len(self.body_lines) - 1
        for body_id, line in enumerate(self.body_lines):
            this_abs = self.n_vars + body_id
            for fi in line.fanins:
                if fi < 0 or fi > max_abs_index or fi >= this_abs:
                    raise ValueError(f"Non-topological fanin {fi} at body line {body_id}")

    def dependency_closure(self, selected_body_ids: Iterable[int]) -> Set[int]:
        closure: Set[int] = set()
        stack = list(selected_body_ids)

        while stack:
            body_id = stack.pop()
            if body_id in closure or body_id < 0 or body_id >= len(self.body_lines):
                continue

            closure.add(body_id)
            for fi in self.body_lines[body_id].fanins:
                if fi >= self.n_vars:
                    dep_id = fi - self.n_vars
                    if dep_id not in closure:
                        stack.append(dep_id)

        return closure

    def write_reduced_library(self, selected_output_ids: Sequence[int], output_path: str) -> Path:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected = sorted(set(selected_output_ids))
        if not selected:
            raise ValueError("Cannot build reduced library with empty selection.")

        # Compute full dependency closure
        closure = self.dependency_closure(selected)
        ordered_old_ids = sorted(closure)

        # Build absolute ID remap dictionary
        abs_id_remap: Dict[int, int] = {i: i for i in range(self.n_vars)}
        for new_body_id, old_body_id in enumerate(ordered_old_ids):
            abs_id_remap[self.n_vars + old_body_id] = self.n_vars + new_body_id

        selected_set = set(selected)
        out_lines = [
            f"{self.genlib_path}",
            f"{self.n_vars}",
            f"{len(selected)}",
            f"{self.n_vars + len(ordered_old_ids)}"
        ]

        for old_body_id in ordered_old_ids:
            old_line = self.body_lines[old_body_id]
            remapped_fanins = [
                fi if fi < self.n_vars else abs_id_remap[fi] 
                for fi in old_line.fanins
            ]
            prefix = "* " if old_body_id in selected_set else ""
            fanins_str = " ".join(str(f) for f in remapped_fanins)
            line_str = f"{prefix}{old_line.gate_name} {fanins_str}".strip()
            out_lines.append(line_str)

        output_path.write_text("\n".join(out_lines) + "\n")
        return output_path