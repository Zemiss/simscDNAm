"""同类型训练缺失掩码和联合秩映射。"""
from pathlib import Path
import gc, hashlib, json, uuid, shutil
import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA

from scipy.stats import rankdata
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

def sample_mask_rows(training_labels, fit_rows, candidate_labels, seed):
    rng = np.random.default_rng(seed)
    donors = np.empty(len(candidate_labels), dtype=int)
    for level in np.unique(candidate_labels):
        pool = fit_rows[training_labels[fit_rows] == level]
        if not len(pool):
            raise ValueError(f'训练掩码没有细胞类型：{level}')
        rows = np.flatnonzero(candidate_labels == level)
        donors[rows] = rng.choice(pool, len(rows), replace=True)
    return donors

def apply_training_mask(candidate, donor_values):
    if candidate.shape != donor_values.shape:
        raise ValueError('生成矩阵和训练掩码形状不一致')
    return np.where(np.isfinite(donor_values), candidate, np.nan).astype(np.float32)


def write_training_mask_h5ad(raw_path, output_path, matrix, training_labels, fit_rows,
                             seed=42, feature_block=4096):
    """同类型fit细胞整行缺失掩码；所有有限生成值和特征轴保持原样。"""
    raw_path, output_path = Path(raw_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'输出已存在：{output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(raw_path, backed='r')
    partial = output_path.with_name(output_path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        if matrix.shape[1] != source.n_vars or len(training_labels) != matrix.shape[0]:
            raise ValueError('训练掩码与生成矩阵的区域轴或行标签不一致')
        labels = source.obs['Subclass'].astype(str).to_numpy()
        donors = sample_mask_rows(np.asarray(training_labels), np.asarray(fit_rows), labels, seed)
        obs = source.obs.copy()
        obs['mask_donor_row'] = donors
        obs['generator'] = 'CVAE＋训练掩码'
        shell = ad.AnnData(sp.csr_matrix(source.shape, dtype=np.float32), obs=obs, var=source.var.copy())
        shell.uns['postprocessing'] = {'method': 'training_mask', 'mask_seed': seed,
            'mask_reference': 'same Subclass, internal fit cells only', 'model_unchanged': True}
        shell.write_h5ad(partial, compression='gzip')
        observed = 0
        with h5py.File(partial, 'r+') as handle:
            del handle['X']
            output = handle.create_dataset('X', shape=source.shape, dtype='float32',
                chunks=(min(8, source.n_obs), min(feature_block, source.n_vars)),
                compression='gzip', compression_opts=1, shuffle=True, fillvalue=np.nan)
            output.attrs['encoding-type'], output.attrs['encoding-version'] = 'array', '0.2.0'
            for start in range(0, source.n_vars, feature_block):
                stop = min(start + feature_block, source.n_vars)
                candidate = source.X[:, start:stop]
                candidate = np.asarray(candidate.toarray() if hasattr(candidate, 'toarray') else candidate, dtype=np.float32)
                donor = np.asarray(matrix[np.ix_(donors, np.arange(start, stop))], dtype=np.float32)
                values = apply_training_mask(candidate, donor)
                np.testing.assert_array_equal(np.isfinite(values), np.isfinite(donor))
                np.testing.assert_array_equal(values[np.isfinite(values)], candidate[np.isfinite(values)])
                observed += int(np.isfinite(values).sum())
                output[:, start:stop] = values
        partial.replace(output_path)
        summary = {'方法': 'CVAE_training_mask', '数据': str(output_path), '掩码种子': seed,
            '细胞数': source.n_obs, '区域数': source.n_vars, '观测率': observed / np.prod(source.shape),
            '仅使用同类型内部训练掩码': True, '有限生成值未改变': True}
        (output_path.parent/'mask_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary
    finally:
        source.file.close()
        if partial.exists():
            partial.unlink()


def gaussian_ranks(values):
    """有限值转平均秩正态分数，缺失中心填0；各非恒定列方差标准化。"""
    finite=np.isfinite(values)
    counts=finite.sum(0)
    ranks=rankdata(np.where(finite,values,np.nan),axis=0,method='average',nan_policy='omit')
    quantiles=(ranks-0.5)/np.maximum(counts,1)
    scores=ndtri(np.clip(quantiles,1e-6,1-1e-6))
    return standardized_scores(scores,finite)

def standardized_scores(values,observed):
    counts=observed.sum(0)
    mean=np.divide(np.sum(values,where=observed,axis=0,dtype=np.float64),counts,
                   out=np.zeros(values.shape[1]),where=counts>0)
    centered=np.where(observed,values-mean,0)
    deviation=np.sqrt(np.sum(centered**2,axis=0)/max(len(values)-1,1))
    result=np.divide(centered,deviation,out=np.zeros_like(centered),where=deviation>1e-7)
    return result.astype(np.float32)

def symmetric_power(matrix,power):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    floor=max(float(values.max())*1e-5,1e-8)
    return (vectors*np.maximum(values,floor)**power)@vectors.T

def fit_rank_factors(reference,n_components):
    scores=gaussian_ranks(reference)
    pca=PCA(n_components=n_components,svd_solver='randomized',iterated_power=3,random_state=42)
    factors=pca.fit_transform(scores)
    gram=factors.T.astype(np.float64)@factors.astype(np.float64)
    centered=scores-pca.mean_
    # 所有区域都走同一回归；不按复制配对表构造或重写任何列。
    basis=np.linalg.solve(gram+np.eye(n_components)*max(np.trace(gram)/n_components,1)*1e-7,
                          factors.T@centered).T.astype(np.float32)
    covariance=(gram/max(len(reference)-1,1)).astype(np.float64)
    return {'basis':basis,'covariance':covariance,'mean':pca.mean_.astype(np.float32),
            'explained_variance':float(pca.explained_variance_ratio_.sum())}

def shared_rank_projection(source_scores,observed,model,n_components):
    q=min(n_components,model['basis'].shape[1],len(source_scores)-1)
    basis=model['basis'][:,:q]
    coordinates=(source_scores-model['mean'])@basis
    coordinates=coordinates@np.linalg.pinv(basis.T@basis,rcond=1e-5)
    coordinates-=coordinates.mean(0)
    covariance=coordinates.T.astype(np.float64)@coordinates.astype(np.float64)/max(len(coordinates)-1,1)
    transform=symmetric_power(covariance,-0.5)@symmetric_power(model['covariance'][:q,:q],0.5)
    transported=(coordinates@transform).astype(np.float32)
    shared=transported@basis.T+model['mean']
    return standardized_scores(shared,observed)

def quantile_from_sorted(reference_sorted,reference_counts,scores,original):
    """逐列显式经验分位数匹配；模型分数只决定细胞间排序。"""
    observed=np.isfinite(original)
    counts=observed.sum(0)
    eligible=(reference_counts>0)&(counts>0)
    result=original.copy()
    if not eligible.any():
        return result
    ranks=rankdata(np.where(observed[:,eligible],scores[:,eligible],np.nan),axis=0,
                   method='average',nan_policy='omit')
    quantiles=(ranks-0.5)/counts[eligible]
    positions=np.nan_to_num(quantiles,nan=0.5)*reference_counts[eligible]-0.5
    positions=np.clip(positions,0,reference_counts[eligible]-1)
    lower=np.floor(positions).astype(int)
    upper=np.minimum(lower+1,reference_counts[eligible]-1)
    ordered=reference_sorted[:,eligible]
    left=np.take_along_axis(ordered,lower,axis=0)
    right=np.take_along_axis(ordered,upper,axis=0)
    mapped=left+(right-left)*(positions-lower)
    result[:,eligible]=np.where(observed[:,eligible],mapped,np.nan)
    return result

def map_class(original,reference_sorted,reference_counts,model,n_components,eta):
    if len(original)<3:
        return original.copy()
    source=gaussian_ranks(original)
    observed=np.isfinite(original)
    if eta:
        shared=shared_rank_projection(source,observed,model,n_components)
    output=np.empty_like(original,dtype=np.float32)
    for start in range(0,original.shape[1],4096):
        stop=min(start+4096,original.shape[1])
        score=source[:,start:stop] if not eta else (1-eta)*source[:,start:stop]+eta*shared[:,start:stop]
        output[:,start:stop]=quantile_from_sorted(reference_sorted[:,start:stop],reference_counts[start:stop],
            score,original[:,start:stop])
    assert np.array_equal(np.isnan(output),np.isnan(original))
    return output



def write_json(path,payload):
    """完成标记和运行签名在写完后原子替换。"""
    path=Path(path)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def feature_axis_hash(names):
    return hashlib.sha256(json.dumps(list(map(str,names)),ensure_ascii=False).encode()).hexdigest()


def copy_checkpoint(source,destination):
    destination=Path(destination)
    temporary=destination.with_name(destination.name+'.partial')
    shutil.copy2(source,temporary)
    temporary.replace(destination)


def fit_joint_mapping(matrix,labels,fit_rows,validation_rows,feature_names,output_dir,components=32):
    """只在fit真实值上拟合；内部验证只提供类型样本数，决定小样本回退和维数。"""
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    labels=np.asarray(labels,dtype=str)
    fit_rows,validation_rows=np.asarray(fit_rows,dtype=int),np.asarray(validation_rows,dtype=int)
    if np.intersect1d(fit_rows,validation_rows).size:
        raise ValueError('fit和内部验证细胞不能重叠')
    if len(feature_names)!=matrix.shape[1] or len(labels)!=matrix.shape[0]:
        raise ValueError('训练矩阵、标签或区域轴不一致')
    identity={'区域轴SHA256':feature_axis_hash(feature_names),'矩阵形状':list(matrix.shape),
              'fit行':fit_rows.tolist(),'验证行':validation_rows.tolist(),'标签':labels.tolist(),
              '因子上限':int(components),'投影强度':1.0}
    identity_path=output_dir/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8'))!=identity:
        raise ValueError('映射运行签名变化，请使用新输出目录')
    if not identity_path.exists():
        write_json(identity_path,identity)
    records=[]
    for index,level in enumerate(np.unique(labels[validation_rows])):
        rows=fit_rows[labels[fit_rows]==level]
        count=int(np.sum(labels[validation_rows]==level))
        record={'细胞类型':str(level),'fit细胞数':len(rows),'内部验证细胞数':count,
                '文件前缀':f'model_{index:03d}'}
        if len(rows)<5 or count<3:
            record['状态']='小样本保留原生成值'
            records.append(record)
            continue
        prefix=output_dir/record['文件前缀']
        model_path=prefix.with_suffix('.npz')
        sorted_path=prefix.with_name(prefix.name+'_sorted.npy')
        if not (model_path.exists() and sorted_path.exists()):
            reference=np.array(matrix[rows],dtype=np.float32)
            counts=np.isfinite(reference).sum(0).astype(np.int32)
            q=min(components,len(rows)-1,count-1)
            with threadpool_limits(limits=8):
                model=fit_rank_factors(reference,q)
            temporary_sorted=sorted_path.with_name(sorted_path.name+'.partial')
            temporary_model=model_path.with_name(model_path.name+'.partial')
            with temporary_sorted.open('wb') as stream:
                np.save(stream,np.sort(np.where(np.isfinite(reference),reference,np.inf),axis=0))
            temporary_sorted.replace(sorted_path)
            with temporary_model.open('wb') as stream:
                np.savez(stream,**model,reference_counts=counts)
            temporary_model.replace(model_path)
            del reference,model
            gc.collect()
        with np.load(model_path) as model:
            record.update({'状态':'已拟合','因子数':int(model['basis'].shape[1]),
                           '秩空间解释方差':float(model['explained_variance'])})
        records.append(record)
        print('联合秩映射拟合：',record,flush=True)
    metadata={**identity,'方法':'simscDNAm','版本':'1.0','类型明细':records}
    write_json(output_dir/'metadata.json',metadata)
    return metadata


def apply_joint_mapping(masked_path,output_path,models_dir):
    """固定联合投影强度1并匹配fit经验分位数；所有列遵循同一规则。"""
    masked_path,output_path,models_dir=Path(masked_path),Path(output_path),Path(models_dir)
    if output_path.exists():
        raise FileExistsError(f'输出已存在：{output_path}')
    metadata_path=models_dir/'metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    original=ad.read_h5ad(masked_path)
    if feature_axis_hash(original.var_names)!=metadata['区域轴SHA256']:
        raise ValueError('生成数据的区域名称或顺序与映射不一致')
    values=original.X.toarray() if sp.issparse(original.X) else original.X.copy()
    labels=original.obs['Subclass'].astype(str).to_numpy()
    details=[]
    for record in metadata['类型明细']:
        rows=np.flatnonzero(labels==record['细胞类型'])
        if record['状态']!='已拟合' or len(rows)<3:
            continue
        prefix=models_dir/record['文件前缀']
        with np.load(prefix.with_suffix('.npz')) as saved:
            model={key:saved[key].copy() for key in ['basis','covariance','mean']}
            counts=saved['reference_counts'].copy()
        ordered=np.load(prefix.with_name(prefix.name+'_sorted.npy'),mmap_mode='r')
        with threadpool_limits(limits=8):
            values[rows]=map_class(values[rows],ordered,counts,model,metadata['因子上限'],1.)
        details.append({'细胞类型':record['细胞类型'],'映射细胞数':len(rows)})
        del model,ordered
        gc.collect()
        print('联合秩映射应用：',details[-1],flush=True)
    source_values=original.X.toarray() if sp.issparse(original.X) else original.X
    np.testing.assert_array_equal(np.isnan(values),np.isnan(source_values))
    if np.nanmin(values)<0 or np.nanmax(values)>1:
        raise ValueError('主模型生成值超出0–1')
    result=ad.AnnData(values,obs=original.obs.copy(),var=original.var.copy(),uns=dict(original.uns))
    result.obs['generator']='simscDNAm'
    result.uns['simscDNAm']={'方法':'simscDNAm','版本':'1.0',
        '流程':'CVAE + training mask + joint rank mapping','因子上限':metadata['因子上限'],
        '投影强度':1.,'fit细胞数':len(metadata['fit行']),
        '映射参数SHA256':hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        '未使用复制配对重写输出':True}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    partial=output_path.with_suffix('.h5ad.partial')
    result.write_h5ad(partial,compression='gzip',compression_opts=1)
    partial.replace(output_path)
    report={'方法':'simscDNAm','状态':'完成','总细胞数':len(labels),
            '映射细胞数':sum(item['映射细胞数'] for item in details),'类型明细':details,
            '数据':str(output_path),'区域数':result.n_vars,'NaN位置保持':True}
    write_json(output_path.parent/'alignment_summary.json',report)
    return report


def finalize_model(dataset_root,matrix,training_labels,fit_rows,validation_rows,names,masked_path,checkpoint_path):
    model_root=Path(dataset_root)/'simscDNAm'
    mapped_path=model_root/'data.h5ad'
    models_dir=model_root/'alignment'
    if not mapped_path.exists():
        fit_joint_mapping(matrix,training_labels,fit_rows,validation_rows,names,models_dir)
        apply_joint_mapping(masked_path,mapped_path,models_dir)
    model_root.mkdir(parents=True,exist_ok=True)
    if not (model_root/'model.pt').exists():
        copy_checkpoint(checkpoint_path,model_root/'model.pt')
    return mapped_path
