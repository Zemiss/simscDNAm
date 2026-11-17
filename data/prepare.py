"""原始 H5AD 的共用预处理命令行。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from data.paths import DATA_ROOT, OUTPUTS, check_output, require_environment
from data.preprocessing import prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description='质控、分层划分并追加 1000 个真实复制区域')
    parser.add_argument('--dataset', default='liu2023/full', help='数据集标识')
    parser.add_argument('--inputs', nargs='+', type=Path, help='一个或多个原始 H5AD')
    parser.add_argument('--output', type=Path, help='新的预处理输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test-fraction', type=float, default=.2, help='留出比例')
    args = parser.parse_args(argv)
    require_environment()
    paths = args.inputs or [DATA_ROOT / 'Liu-2023_nature' / accession / f'{accession}.h5ad'
        for accession in ('GSE213221','GSE215213','GSE215342','GSE230224','GSE230226')]
    for path in paths:
        path.resolve().relative_to(DATA_ROOT.resolve())
    output = check_output(args.output or OUTPUTS / 'preprocessing' / args.dataset)
    print(json.dumps(prepare_dataset(paths, output, args.seed, args.test_fraction), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
