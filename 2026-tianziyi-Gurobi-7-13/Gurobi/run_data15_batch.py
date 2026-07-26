"""Run all eight 15-order instances with mip_revised.py in parallel."""

from pathlib import Path

import run_data20_batch as batch


ROOT = Path(__file__).resolve().parent
batch.DATA_ROOT = ROOT / "数据" / "15"
batch.RESULTS_ROOT = ROOT / "批量求解结果_数据15_20260723"


if __name__ == "__main__":
    batch.main()
