"""
CLI Runner for Data Preparation and Feature Pipeline.
Executes PySpark window transformations (or vectorized fallback) and saves featured Parquet.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.pyspark_pipeline import execute_etl
from src.utils.config import load_config


def main():
    print("=" * 60)
    print("  RETAIL DEMAND ETL & WINDOW FEATURE PIPELINE")
    print("=" * 60)
    config = load_config()
    execute_etl(config)
    print("\n[ETL Pipeline] Finished successfully.")


if __name__ == "__main__":
    main()
