"""项目路径、服务器输出约束与运行环境检查。"""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
SERVER = sys.platform != 'win32'
DATA_ROOT = Path('/home/sccasimp/data/xb/simscDNAm') if SERVER else PROJECT / 'data/raw'
OUTPUTS = Path('/home/sccasimp/data/xb/outputs/simscDNAm') if SERVER else PROJECT / 'outputs'


def check_output(path):
    path = Path(path).expanduser().resolve()
    if SERVER:
        path.relative_to(OUTPUTS.resolve())
    return path


def require_environment():
    if Path(sys.prefix).name.lower() != 'simscdnam':
        raise RuntimeError(f'请使用 simscDNAm Python 环境，当前为 {sys.executable}')
