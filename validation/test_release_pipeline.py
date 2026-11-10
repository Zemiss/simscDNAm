"""正式主模型通过Jupyter执行映射，核对fit隔离、输出范围和联合约束。"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_mapping_runs_in_jupyter_and_keeps_contract(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'simscDNAm/train.ipynb'
    assert path.is_file(), '正式主模型入口缺失'
    source=nbformat.read(path,as_version=4)
    definitions=[cell for cell in source.cells if cell.cell_type=='code' and cell.metadata.get('definitions')]
    check=nbformat.v4.new_code_cell('''
from scipy.stats import wasserstein_distance
rng=np.random.default_rng(42)
names=np.array(['A','B','A_copy','C','D'])
labels=np.array(['Main']*12+['Rare']*3)
reference=np.array([[0,0,0,.9,.1],[.1,.9,.1,.8,.2],[.2,.1,.2,.7,.3],
                    [.7,.8,.7,.3,.7],[.9,.2,.9,.2,.8],[1,1,1,.1,.9]]*2+
                   [[.2,.3,.2,.4,.5]]*3,dtype=np.float32)
fit=np.array([0,1,2,3,4,5,12,13])
validation=np.array([6,7,8,9,10,11,14])
metadata=fit_joint_mapping(reference,labels,fit,validation,names,TMP/'mapping',components=3)
changed=reference.copy()
changed[validation]=999
second=fit_joint_mapping(changed,labels,fit,validation,names,TMP/'mapping_changed',components=3)
first_model=np.load(TMP/'mapping/model_000.npz')
second_model=np.load(TMP/'mapping_changed/model_000.npz')
np.testing.assert_array_equal(first_model['basis'],second_model['basis'])
np.testing.assert_array_equal(first_model['reference_counts'],second_model['reference_counts'])
assert metadata['类型明细'][1]['状态']=='小样本保留原生成值'
generated=np.array([[.4,.3,.41,.6,.4],[.45,.4,.46,.55,.45],[.5,.6,.51,.5,.5],
                    [.55,.7,.54,.45,.55],[.6,.8,.59,.4,.6],[.65,.9,.64,.35,.65],
                    [.2,.3,.21,.4,.5]],dtype=np.float32)
generated[0,4]=np.nan
obs=pd.DataFrame({'Subclass':['Main']*6+['Rare'],'mask_donor_row':[0,1,2,3,4,5,12]},
                 index=[f's{i}' for i in range(7)])
item=ad.AnnData(generated,obs=obs,var=pd.DataFrame(index=names))
item.write_h5ad(TMP/'masked.h5ad')
before=(TMP/'masked.h5ad').read_bytes()
report=apply_joint_mapping(TMP/'masked.h5ad',TMP/'simscDNAm/data.h5ad',TMP/'mapping')
result=ad.read_h5ad(TMP/'simscDNAm/data.h5ad')
assert result.uns['simscDNAm']['方法']=='simscDNAm'
assert result.uns['simscDNAm']['投影强度']==1.
np.testing.assert_array_equal(np.isnan(result.X),np.isnan(generated))
np.testing.assert_array_equal(result.X[:6,0],result.X[:6,2])
np.testing.assert_array_equal(result.X[-1],generated[-1])
assert result.obs_names.equals(item.obs_names) and result.var_names.equals(item.var_names)
np.testing.assert_array_equal(result.obs['mask_donor_row'],item.obs['mask_donor_row'])
assert np.nanmin(result.X)>=0 and np.nanmax(result.X)<=1
expected=np.interp((np.arange(6)+.5)/6,(np.arange(6)+.5)/6,np.sort(reference[fit[:6],0]))
np.testing.assert_allclose(np.sort(result.X[:6,0]),expected,atol=1e-6)
assert (TMP/'masked.h5ad').read_bytes()==before
try:
    apply_joint_mapping(TMP/'masked.h5ad',TMP/'simscDNAm/data.h5ad',TMP/'mapping')
except FileExistsError:
    pass
else:
    raise AssertionError('不得覆盖已有映射输出')
bad=item[:,::-1].copy()
bad.write_h5ad(TMP/'bad.h5ad')
try:
    apply_joint_mapping(TMP/'bad.h5ad',TMP/'bad_output.h5ad',TMP/'mapping')
except ValueError:
    pass
else:
    raise AssertionError('区域顺序错误应拒绝')
''')
    notebook=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell('from pathlib import Path\nTMP=Path('+repr(str(tmp_path))+')'),
        *definitions,check],metadata=source.metadata)
    NotebookClient(notebook,kernel_name='simscdnam',timeout=120,
                   resources={'metadata':{'path':str(root)}}).execute()


def test_four_method_defaults_and_integrated_main():
    root=Path(__file__).resolve().parents[1]
    import yaml
    config=yaml.safe_load((root/'evaluate/config.yaml').read_text(encoding='utf-8'))
    assert list(config['synthetic_h5ad'])==['simscDNAm','FittedMarginal','simATAC','simCAS']
    path=root/'simscDNAm/train.ipynb'
    assert path.is_file(), '正式主模型入口缺失'
    notebook=nbformat.read(path,as_version=4)
    source='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
    assert source.index('write_training_mask_h5ad(raw_path, masked_path') < source.index('mapped_path = finalize_model(')
    assert '"方法": "simscDNAm"' in source
