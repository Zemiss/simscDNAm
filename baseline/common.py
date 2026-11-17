"""基线命令行的共用输入与输出参数。"""
import argparse
from pathlib import Path

from data.paths import OUTPUTS, check_output, require_environment


def parse_arguments(method, argv=None, spatial_diffusion=False):
    parser = argparse.ArgumentParser(description=f'{method} 基线训练与生成')
    parser.add_argument('--dataset', default='liu2023/full', help='数据集标识，默认 liu2023/full')
    parser.add_argument('--training', type=Path, help='预处理后的训练 H5AD')
    parser.add_argument('--target', type=Path, help='仅提供目标 Subclass 配额的 H5AD')
    parser.add_argument('--output', type=Path, help='本方法独立输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    if spatial_diffusion:
        parser.add_argument('--mode', choices=('full', 'pilot'), default='full', help='正式训练或独立试运行')
        parser.add_argument('--device', default='cuda:0', help='CUDA 设备')
    else:
        parser.add_argument('--reference', type=Path, help='可选：逐元素核对已有生成结果')
        parser.add_argument('--replay-root', type=Path, help='可选：核对并重放既有历史中间结果')
    args = parser.parse_args(argv)
    require_environment()
    prepared = OUTPUTS / 'preprocessing' / args.dataset
    args.training = (args.training or prepared / 'TrainingReal.h5ad').expanduser().resolve()
    args.target = (args.target or prepared / 'HeldoutReal.h5ad').expanduser().resolve()
    args.output = check_output(args.output or OUTPUTS / 'baseline' / args.dataset / method)
    return args
