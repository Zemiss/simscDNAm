"""原始甲基化数据质控、分层划分及训练前追加复制区域。"""
from pathlib import Path
from contextlib import ExitStack
import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data.training import inspect_inputs, materialize_inputs, stratified_group_split
from data.evaluation import region_pairs, validate_real_duplicates


def filter_training_subclasses(training_path, heldout_path, output_dir, min_cells=10):
    """按外层训练集的 Subclass 频数过滤两份真实数据，保留原有行列顺序。"""
    output_dir = Path(output_dir).resolve()
    with ExitStack() as stack:
        training, heldout = [ad.read_h5ad(path, backed='r') for path in (training_path, heldout_path)]
        for item in (training, heldout):
            stack.callback(item.file.close)
        if not training.var_names.equals(heldout.var_names) or len(training.obs_names.intersection(heldout.obs_names)):
            raise ValueError('训练与测试必须具有相同区域轴、相互独立的细胞')
        labels = [item.obs['Subclass'].astype(str) for item in (training, heldout)]
        counts = pd.concat([s.value_counts() for s in labels], axis=1).fillna(0).astype(int).sort_index()
        counts.columns = ['训练细胞数', '测试细胞数']
        counts['保留'] = counts['训练细胞数'] >= min_cells
        counts.index.name = 'Subclass'
        kept = counts.index[counts['保留']]
        masks = [s.isin(kept).to_numpy() for s in labels]
        if not all(mask.any() for mask in masks):
            raise ValueError('过滤后训练或测试集为空')
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, item, mask in zip(('TrainingReal', 'HeldoutReal'), (training, heldout), masks):
            path = output_dir / f'{name}.h5ad'
            if path in (Path(training_path).resolve(), Path(heldout_path).resolve()):
                raise ValueError('过滤结果不能覆盖输入')
            partial = path.with_suffix('.h5ad.partial')
            item[mask].to_memory().write_h5ad(partial, compression='gzip')
            partial.replace(path)
    return counts


def write_expanded_real(path, matrix, rows, metadata, var, pairs, row_block=16):
    """分块导出真实训练或留出数据，保留零值、NaN 和固定的复制对应关系。"""
    copied = var.iloc[pairs['原region索引'].to_numpy()].copy()
    copied.index = pd.Index(pairs['复制region'].to_numpy())
    expanded_var = pd.concat([var, copied])
    obs = metadata.loc[rows, ['cell_id', 'Subclass', 'source_dataset']].copy().set_index('cell_id')
    if obs.empty or not obs.index.is_unique or not expanded_var.index.is_unique:
        raise ValueError('细胞和区域 ID 必须唯一，划分后的数据不能为空')
    path = Path(path)
    partial = path.with_suffix('.h5ad.partial')
    try:
        ad.AnnData(sp.csr_matrix((len(rows), len(expanded_var)), dtype=np.float32),
            obs=obs, var=expanded_var).write_h5ad(partial)
        with h5py.File(partial, 'r+') as handle:
            del handle['X']
            output = handle.create_dataset('X', shape=(len(rows), len(expanded_var)), dtype='float32',
                chunks=(min(row_block, len(rows)), min(4096, len(expanded_var))), compression='gzip')
            output.attrs['encoding-type'], output.attrs['encoding-version'] = 'array', '0.2.0'
            originals = pairs['原region索引'].to_numpy(dtype=int)
            for start in range(0, len(rows), row_block):
                values = np.asarray(matrix[rows[start:start+row_block]], dtype=np.float32)
                output[start:start+len(values)] = np.column_stack([values, values[:, originals]])
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def prepare_dataset(paths, output_dir, seed=42, test_fraction=0.2):
    """按源数据集和 Subclass 分层划分，再向两个真实集合追加同一批 1000 个区域。"""
    paths = [Path(path).resolve() for path in paths]
    output_dir = Path(output_dir).resolve()
    if not paths or not 0 < test_fraction < 1:
        raise ValueError('必须提供原始输入，留出比例必须在 0 与 1 之间')
    if any(path.is_relative_to(output_dir) for path in paths):
        raise ValueError('预处理输出目录不能包含输入文件')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('预处理输出目录非空，请选择新目录，防止覆盖既有划分')
    source_table, names, var = inspect_inputs(paths)
    if len(names) < 1000 or not pd.Index(names).is_unique:
        raise ValueError('原始区域必须唯一且不少于 1000 个')
    if any('__duplicate_' in str(name) for name in names):
        raise ValueError('输入已有复制区域，请使用原始 H5AD，避免重复扩维')
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / 'work'
    work.mkdir()
    matrix = None
    try:
        matrix, metadata, retained, missing_fraction = materialize_inputs(paths, work, names)
        if not metadata['cell_id'].is_unique:
            raise ValueError('合并后的细胞 ID 不唯一，请检查输入是否重复')
        train_rows, heldout_rows = stratified_group_split(metadata, retained, test_fraction, seed)
        indices = np.sort(np.random.default_rng(seed).choice(len(names), 1000, replace=False))
        pairs = pd.DataFrame({'原region索引': indices, '复制region索引': len(names) + np.arange(1000),
            '原region': names[indices],
            '复制region': [f'{names[index]}__duplicate_{i:04d}' for i, index in enumerate(indices)]})
        for filename, rows in [('TrainingReal.h5ad', train_rows), ('HeldoutReal.h5ad', heldout_rows)]:
            destination = output_dir / filename
            write_expanded_real(destination, matrix, rows, metadata, var, pairs)
            item = ad.read_h5ad(destination, backed='r')
            try:
                validate_real_duplicates(item, region_pairs(item))
            finally:
                item.file.close()
        pairs.to_csv(output_dir / 'region_pairs.csv', index=False)
        metadata.loc[train_rows].to_csv(output_dir / 'training_cells.csv', index=False)
        metadata.loc[heldout_rows].to_csv(output_dir / 'heldout_cells.csv', index=False)
        for record in source_table.to_dict('records'):
            path = Path(record['path'])
            if (path.stat().st_size, path.stat().st_mtime_ns) != (record['size'], record['mtime_ns']):
                raise RuntimeError(f'预处理期间原始输入发生变化：{path}')
        report = {'状态': '完成', '随机种子': seed, '留出比例': test_fraction,
            '原始细胞数': len(metadata), '质控保留细胞数': len(retained),
            '训练细胞数': len(train_rows), '留出细胞数': len(heldout_rows),
            '原始区域数': len(names), '复制区域数': 1000, '原始缺失比例': missing_fraction,
            '输入': source_table.to_dict('records'), '输出目录': str(output_dir)}
        (output_dir / 'run_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
    finally:
        if matrix is not None:
            matrix._mmap.close()
