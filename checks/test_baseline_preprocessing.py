"""检查新增基线的数据适配边界，不训练模型。"""
from pathlib import Path
import hashlib,json,tempfile,unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch
import anndata as ad
import numpy as np
import pandas as pd
from data.baseline import prepare_imputed_training,clip_generated_matrix

class BaselinePreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        natural=np.full((4,1000),.2,dtype=np.float32)
        natural[:,0]=[0,np.nan,1,.5]
        natural[:,1]=np.nan
        natural[:,2]=[.2,.4,np.nan,.8]
        names=[f'region_{index}' for index in range(1000)]
        names += [f'region_{index}__duplicate_{index:04d}' for index in range(1000)]
        self.training=self.root/'train.h5ad'
        self.target=self.root/'target.h5ad'
        obs=pd.DataFrame({'Subclass':['A','A','B','B']},index=['t1','t2','t3','t4'])
        ad.AnnData(np.column_stack([natural,natural]),obs=obs,var=pd.DataFrame(index=names)).write_h5ad(self.training)
        # 留出数值故意与训练范围不符；适配只能读取其标签和区域轴。
        ad.AnnData(np.full((2,2000),99,dtype=np.float32),obs=pd.DataFrame({'Subclass':['A','B']},index=['h1','h2']),var=pd.DataFrame(index=names)).write_h5ad(self.target)
    def tearDown(self):self.temporary.cleanup()
    def test_training_only_medians_preserve_zero_and_duplicate_columns(self):
        before=hashlib.sha256(self.training.read_bytes()).hexdigest()
        path,target,report=prepare_imputed_training(self.training,self.target,self.root/'cache')
        filled=ad.read_h5ad(path)
        np.testing.assert_array_equal(filled.X[:,0],np.array([0,.5,1,.5],dtype=np.float32))
        np.testing.assert_allclose(filled.X[:,1],.2)
        self.assertEqual(filled.X[2,2],np.float32(.4))
        np.testing.assert_array_equal(filled.X[:,:1000],filled.X[:,1000:])
        self.assertTrue(np.isfinite(filled.X).all())
        self.assertEqual(target['Subclass'].tolist(),['A','B'])
        self.assertEqual(before,hashlib.sha256(self.training.read_bytes()).hexdigest())
    def test_cache_rejects_changed_training(self):
        prepare_imputed_training(self.training,self.target,self.root/'cache')
        item=ad.read_h5ad(self.training)
        item.X[0,3]=.7;item.X[0,1003]=.7
        item.write_h5ad(self.training)
        with self.assertRaises(ValueError):prepare_imputed_training(self.training,self.target,self.root/'cache')
    def test_parallel_cold_cache_publishes_complete_files(self):
        from data.baseline import read_training
        barrier=Barrier(2)
        def synchronized_read(path):
            result=read_training(path)
            barrier.wait(timeout=10)
            return result
        with patch('data.baseline.read_training',side_effect=synchronized_read),ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(prepare_imputed_training,self.training,self.target,self.root/'cache') for _ in range(2)]
            results=[future.result(timeout=20) for future in futures]
        self.assertEqual(results[0][0],results[1][0])
        self.assertTrue(np.isfinite(ad.read_h5ad(results[0][0]).X).all())
        self.assertTrue(np.isfinite(np.load(self.root/'cache/training_medians.npy')).all())
        self.assertEqual(json.loads((self.root/'cache/imputation.json').read_text(encoding='utf-8'))['状态'],'完成')
    def test_output_clip_preserves_in_range_values_without_row_or_column_change(self):
        values=np.array([[-3,0,.25,1,8]],dtype=np.float32)
        original=values.copy()
        clipped,report=clip_generated_matrix(values)
        np.testing.assert_array_equal(clipped,np.array([[0,0,.25,1,1]],dtype=np.float32))
        np.testing.assert_array_equal(values,original)
        self.assertEqual(clipped.dtype,np.dtype('float32'))
    def test_clip_rejects_nonfinite_instead_of_inventing_values(self):
        for value in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(ValueError):clip_generated_matrix(np.array([[value]],dtype=np.float32))

if __name__=='__main__':unittest.main()
