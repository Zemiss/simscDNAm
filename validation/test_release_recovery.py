"""正式入口的恢复与原始／扩维留出输入回归检查。"""
from pathlib import Path

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def functions():
    path=Path(__file__).resolve().parents[1]/'simscDNAm/train.ipynb'
    notebook=nbformat.read(path,as_version=4)
    scope={}
    for cell in notebook.cells:
        if cell.cell_type=='code' and cell.metadata.get('definitions'):
            exec(compile(cell.source,str(path),'exec'),scope)
    return scope


@pytest.fixture
def training():
    rng=np.random.default_rng(42)
    values=rng.uniform(0,1,(12,1002)).astype(np.float32)
    values[0,2]=np.nan
    names=[f'r{i}' for i in range(1002)]
    names += [f'r{i}__duplicate_{i:04d}' for i in range(1000)]
    item=ad.AnnData(np.c_[values,values[:,:1000]],
        obs=pd.DataFrame({'Subclass':['A']*6+['B']*6},index=[f't{i}' for i in range(12)]),
        var=pd.DataFrame(index=names))
    pairs=pd.DataFrame({'原region索引':np.arange(1000),'复制region索引':np.arange(1002,2002)})
    return item,pairs


@pytest.mark.parametrize('expanded',[False,True])
def test_heldout_accepts_raw_and_expanded_axis(functions,training,tmp_path,expanded):
    item,pairs=training
    target=item[:4].copy()
    target.obs_names=[f'h{i}' for i in range(4)]
    if not expanded:
        target=target[:,:1002].copy()
    target.write_h5ad(tmp_path/'target.h5ad')
    functions['expanded_heldout'](tmp_path/'target.h5ad',tmp_path/'result.h5ad',item,pairs)
    result=ad.read_h5ad(tmp_path/'result.h5ad')
    np.testing.assert_array_equal(result.X,item.X[:4])
    assert result.obs_names.equals(target.obs_names)


def test_interrupted_parameter_write_never_commits_corrupt_cache(functions,tmp_path,monkeypatch):
    matrix=np.random.default_rng(42).uniform(0,1,(12,5)).astype(np.float32)
    def interrupted(path,*args,**kwargs):
        if hasattr(path,'write'):
            path.write(b'truncated')
        else:
            Path(path).write_bytes(b'truncated')
        raise OSError('模拟写入中断')
    with monkeypatch.context() as patch:
        patch.setattr(np,'savez',interrupted)
        with pytest.raises(OSError,match='模拟写入中断'):
            functions['fit_joint_mapping'](matrix,np.array(['A']*12),np.arange(8),np.arange(8,12),
                                           list('abcde'),tmp_path/'mapping',components=3)
    assert not (tmp_path/'mapping/model_000.npz').exists()
    functions['fit_joint_mapping'](matrix,np.array(['A']*12),np.arange(8),np.arange(8,12),
                                   list('abcde'),tmp_path/'mapping',components=3)
    with np.load(tmp_path/'mapping/model_000.npz') as model:
        assert model['basis'].shape==(5,3)


def test_reuse_rejects_changed_training_and_changed_quotas(functions,training,tmp_path):
    item,_=training
    item.write_h5ad(tmp_path/'training.h5ad')
    np.save(tmp_path/'values.npy',item.X)
    pd.DataFrame({'cell_id':item.obs_names,'Subclass':item.obs['Subclass'].to_numpy()}).to_csv(
        tmp_path/'audit.csv',index=False)
    target=item[[0,1,6,7]].copy()
    target.obs_names=['s0','s1','s2','s3']
    target.write_h5ad(tmp_path/'masked.h5ad')
    target[:,:1002].write_h5ad(tmp_path/'target.h5ad')
    args=[tmp_path/'training.h5ad',tmp_path/'values.npy',tmp_path/'audit.csv',
          tmp_path/'masked.h5ad',tmp_path/'target.h5ad']
    validate=functions['validate_pretrained_inputs']
    validate(*args)
    damaged=item.copy()
    damaged.X[0,0]+=.1
    damaged.write_h5ad(tmp_path/'training.h5ad')
    with pytest.raises(ValueError,match='训练'):
        validate(*args)
    item.write_h5ad(tmp_path/'training.h5ad')
    target.obs['Subclass']=['A','A','A','B']
    target[:,:1002].write_h5ad(tmp_path/'target.h5ad')
    with pytest.raises(ValueError,match='配额'):
        validate(*args)


def test_interrupted_checkpoint_copy_can_resume(functions,tmp_path,monkeypatch):
    import shutil
    source,destination=tmp_path/'source.pt',tmp_path/'model.pt'
    source.write_bytes(b'complete checkpoint')
    copy=functions['copy_checkpoint']
    def interrupted(source_path,destination_path):
        Path(destination_path).write_bytes(b'partial')
        raise OSError('模拟复制中断')
    with monkeypatch.context() as patch:
        patch.setattr(shutil,'copy2',interrupted)
        with pytest.raises(OSError,match='模拟复制中断'):
            copy(source,destination)
    assert not destination.exists()
    copy(source,destination)
    assert destination.read_bytes()==source.read_bytes()
