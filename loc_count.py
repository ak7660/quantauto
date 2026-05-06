"""
loc_count.py — Lines-of-code counter for the QuantAuto repository.

Methodology
-----------
* **Python source files** (``*.py``): every ``.py`` file under the repo root
  (excluding the ``.git`` directory) is read line-by-line.  Each line is
  classified as blank (empty or only whitespace), a comment (first non-whitespace
  character is ``#``), or code.
* **Jupyter notebooks** (``*.ipynb``): only *code* cells are counted; markdown
  cells and all cell *outputs* are ignored.  Lines inside code cells follow the
  same blank / comment / code classification as plain Python files.
* **Everything else** (data files, images, config, docs, …) is excluded.

Run
---
    python loc_count.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileStats:
    path: str
    total: int = 0
    blank: int = 0
    comment: int = 0
    code: int = 0


@dataclass
class Summary:
    file_stats: List[FileStats] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(f.total for f in self.file_stats)

    @property
    def blank(self) -> int:
        return sum(f.blank for f in self.file_stats)

    @property
    def comment(self) -> int:
        return sum(f.comment for f in self.file_stats)

    @property
    def code(self) -> int:
        return sum(f.code for f in self.file_stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_lines(lines: List[str]) -> tuple[int, int, int]:
    """Return (blank, comment, code) counts for a list of source lines."""
    blank = comment = code = 0
    for raw in lines:
        stripped = raw.rstrip("\n").strip()
        if not stripped:
            blank += 1
        elif stripped.startswith("#"):
            comment += 1
        else:
            code += 1
    return blank, comment, code


def _collect_files(root: str):
    """Yield absolute file paths under *root*, skipping hidden directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            yield os.path.join(dirpath, fname)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

def count_python(root: str) -> Summary:
    summary = Summary()
    for fpath in sorted(_collect_files(root)):
        if not fpath.endswith(".py"):
            continue
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        blank, comment, code = _classify_lines(lines)
        summary.file_stats.append(
            FileStats(
                path=os.path.relpath(fpath, root),
                total=len(lines),
                blank=blank,
                comment=comment,
                code=code,
            )
        )
    return summary


def count_notebooks(root: str) -> Summary:
    """Count only code-cell lines; outputs and markdown cells are excluded."""
    summary = Summary()
    for fpath in sorted(_collect_files(root)):
        if not fpath.endswith(".ipynb"):
            continue
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            nb = json.load(fh)
        lines: List[str] = []
        for cell in nb.get("cells", []):
            if cell["cell_type"] != "code":
                continue
            src = cell.get("source", [])
            if isinstance(src, list):
                lines.extend(src)
            else:
                lines.extend(src.splitlines(keepends=True))
        blank, comment, code = _classify_lines(lines)
        summary.file_stats.append(
            FileStats(
                path=os.path.relpath(fpath, root),
                total=len(lines),
                blank=blank,
                comment=comment,
                code=code,
            )
        )
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_W = 62
_HDR = f"{'File':<{_COL_W}} {'Total':>6} {'Blank':>6} {'Comment':>8} {'Code':>6}"
_SEP = "-" * (_COL_W + 30)


def _print_section(title: str, summary: Summary) -> None:
    print(f"\n{'=' * (_COL_W + 30)}")
    print(f"  {title}")
    print(f"{'=' * (_COL_W + 30)}")
    print(_HDR)
    print(_SEP)
    for fs in summary.file_stats:
        print(f"{fs.path:<{_COL_W}} {fs.total:>6} {fs.blank:>6} {fs.comment:>8} {fs.code:>6}")
    print(_SEP)
    print(
        f"{'SUBTOTAL':<{_COL_W}} {summary.total:>6} {summary.blank:>6} "
        f"{summary.comment:>8} {summary.code:>6}"
    )


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))

    py_summary = count_python(root)
    nb_summary = count_notebooks(root)

    _print_section("Python source files  (*.py)", py_summary)
    _print_section(
        "Jupyter notebooks  (*.ipynb) — code cells only, outputs excluded",
        nb_summary,
    )

    grand_total = py_summary.total + nb_summary.total
    grand_blank = py_summary.blank + nb_summary.blank
    grand_comment = py_summary.comment + nb_summary.comment
    grand_code = py_summary.code + nb_summary.code

    print(f"\n{'=' * (_COL_W + 30)}")
    print(f"  GRAND TOTAL  (Python + Notebook code cells, no data/outputs)")
    print(f"{'=' * (_COL_W + 30)}")
    print(_HDR)
    print(_SEP)
    print(
        f"{'TOTAL':<{_COL_W}} {grand_total:>6} {grand_blank:>6} "
        f"{grand_comment:>8} {grand_code:>6}"
    )
    print(_SEP)
    print()
    print(f"  Lines of code (excl. blank & comment lines): {grand_code:,}")
    print(f"  Total source lines (incl. blank & comments): {grand_total:,}")
    print()


if __name__ == "__main__":
    main()
