"""训练命令行入口，具体实现位于 model.src。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.src.pipeline import main

if __name__ == "__main__":
    main()
