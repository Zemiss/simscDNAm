"""STADiffuser 非空间配置的固定轮数训练与生成。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.paths import PROJECT, OUTPUTS
from baseline.common import parse_arguments

def run(training_path, target_obs_path, output_dir, seed=42, mode="full", device="cuda:0"):
    RUN_MODE = mode
    TRAINING_PATH, TARGET_OBS_PATH = Path(training_path), Path(target_obs_path)
    OUTPUT_ROOT = Path(output_dir)
    OFFICIAL_COMMIT = "11be35ffce082d5d9cd4bfdcc8f8352fb4a58971"
    OFFICIAL_ROOT = PROJECT / "baseline/STADiffuser/vendor"
    RUNTIME_ROOT = PROJECT / "baseline/STADiffuser/.runtime"
    DEVICE_NAME = device
    SEED = seed
    BATCH_SIZE = 256
    FEATURE_BLOCK = 1024
    import os, sys, json, hashlib, subprocess, time, gc, warnings
    import importlib.metadata as package_metadata
    os.environ['USE_FLAX'] = '0'
    os.environ['USE_TF'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    assert Path(sys.prefix).name.lower() == 'simscdnam', sys.executable
    assert RUN_MODE in {'pilot','full'}
    OUTPUT_ROOT.resolve().relative_to(OUTPUTS.resolve())
    if not (RUNTIME_ROOT / 'diffusers').is_dir():
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(RUNTIME_ROOT),
            'diffusers==0.17.1','huggingface_hub==0.25.2','torch_geometric==2.3.1'],check=True)
    sys.path.insert(0,str(RUNTIME_ROOT))

    SOURCE_HASHES = {'stadiffuser/__init__.py': 'ec6828eb5f418fa1738d7960399b1e530788c7bf95a24f700bf0c52ce791d891', 'stadiffuser/models.py': 'c61eb117fb632528f28c1474b624ce0798522e9feb8b2cd2fd76168366faf9ae', 'stadiffuser/vae.py': '774f7a75da994b532e940cd0f4a549b5e3f1dfdab0b9febc12bb1d00f53eee17', 'stadiffuser/modules.py': '63d399bdfbd10cc1d7f8c50223095c353b0fdc448d50ef1b314752a73e169eea', 'stadiffuser/utils.py': 'd55f404551bdd3bf433a8303359eb187a55e39939d431f2252fbe428a68fc0cb', 'LICENSE.txt': 'fc219795731037e7d2559c433d0e1791f9824d1c60916a450554e3d4d2cd5fab'}

    for relative, expected in SOURCE_HASHES.items():
        path = OFFICIAL_ROOT / relative
        if not path.is_file():
            import urllib.request
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/zhanglabtools/STADiffuser/{OFFICIAL_COMMIT}/{relative}'
            path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'官方源码哈希不一致：{path}')
    sys.path.insert(0,str(OFFICIAL_ROOT))
    import torch
    from torch.nn import functional as F
    from diffusers import DDPMScheduler
    from stadiffuser.vae import SpaAE
    from stadiffuser.models import SpaUNet1DModel
    from stadiffuser.utils import MinMaxNormalize
    torch.set_num_threads(8)
    DEVICE = torch.device(DEVICE_NAME)
    torch.cuda.set_device(DEVICE)
    torch.manual_seed(SEED)
    AE_EPOCHS, DIFFUSION_EPOCHS = (2, 3) if RUN_MODE == 'pilot' else (500, 1000)
    RUN_DIR = OUTPUT_ROOT if RUN_MODE == "full" else OUTPUT_ROOT / "pilot"
    RUN_DIR.mkdir(parents=True,exist_ok=True)
    print({'模式':RUN_MODE,'解释器':sys.executable,'设备':str(DEVICE),'官方提交':OFFICIAL_COMMIT,
           'AE轮数':AE_EPOCHS,'扩散轮数':DIFFUSION_EPOCHS})

    import numpy as np
    import pandas as pd
    import anndata as ad
    import scipy.sparse as sp
    sys.path.insert(0, str(PROJECT))
    from data.evaluation import region_pairs as read_pairs, validate_real_duplicates
    def hash_text(values):
        digest = hashlib.sha256()
        for value in values:
            digest.update(str(value).encode('utf-8')); digest.update(b'\0')
        return digest.hexdigest()


    def fit_medians(matrix, rows, feature_block=1024):
        medians = np.empty(matrix.shape[1],dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for start in range(0,matrix.shape[1],feature_block):
                stop = min(start+feature_block,matrix.shape[1])
                medians[start:stop] = np.nanmedian(matrix[rows,start:stop],axis=0)
        finite = medians[np.isfinite(medians)]
        if not len(finite): raise ValueError('训练集没有有限值')
        medians[~np.isfinite(medians)] = np.median(finite)
        return medians


    def fill_rows(matrix,rows,medians):
        values = np.array(matrix[rows],dtype=np.float32,copy=True)
        missing = ~np.isfinite(values)
        values[missing] = medians[np.nonzero(missing)[1]]
        return values


    def self_loop_edges(count,device):
        return torch.arange(count,device=device,dtype=torch.long).repeat(2,1)


    def synthetic_obs(labels,seed=42):
        labels = np.asarray(labels,dtype=str).copy()
        np.random.default_rng(seed).shuffle(labels)
        return pd.DataFrame({'Subclass':labels,'origin':'Synthetic'},
            index=pd.Index([f'synthetic_{i:06d}' for i in range(1,len(labels)+1)],name='cell_id'))


    def write_generated(path,values,obs,var):
        values = np.asarray(values,dtype=np.float32)
        if values.shape != (len(obs),len(var)) or not np.isfinite(values).all():
            raise ValueError('生成矩阵形状错误或含非有限值')
        partial = Path(str(path)+'.partial')
        ad.AnnData(X=values,obs=obs.copy(),var=var.copy()).write_h5ad(partial,compression='gzip')
        partial.replace(path)


    def predict_noise(model,sample,timesteps,labels):
        return model(sample,timesteps,spatial_coord=None,class_labels=labels).sample

    training_item = ad.read_h5ad(TRAINING_PATH)
    heldout_item = ad.read_h5ad(TARGET_OBS_PATH)
    if not training_item.var_names.equals(heldout_item.var_names):
        raise ValueError("训练和留出区域轴不一致")
    if len(training_item.obs_names.intersection(heldout_item.obs_names)):
        raise ValueError("训练与留出细胞重叠")
    region_pairs = read_pairs(training_item)
    validate_real_duplicates(training_item, region_pairs)
    validate_real_duplicates(heldout_item, region_pairs)
    training = np.asarray(training_item.X.toarray() if sp.issparse(training_item.X) else training_item.X, dtype=np.float32)
    train_labels = training_item.obs["Subclass"].astype(str).to_numpy()
    feature_var = training_item.var.copy()
    if RUN_MODE == "pilot":
        natural = training_item.n_vars - 1000
        selected = np.unique(np.r_[np.arange(min(3096, natural)), region_pairs["原region索引"]])
        selected = np.r_[selected, np.arange(natural, training_item.n_vars)]
        training_item = training_item[:, selected].copy()
        heldout_item = heldout_item[:64, selected].copy()
        rows = np.sort(np.concatenate([g.index.to_numpy()[:4] for _, g in
            pd.DataFrame({"Subclass": train_labels}).groupby("Subclass", sort=True)]))
        training_item = training_item[rows].copy()
        training = training[np.ix_(rows, selected)]
        train_labels = train_labels[rows]
        feature_var = training_item.var.copy()
        region_pairs = read_pairs(training_item)
    target_obs = synthetic_obs(heldout_item.obs["Subclass"].to_numpy(dtype=str), SEED)
    classes = np.sort(np.unique(train_labels))
    assert set(target_obs["Subclass"]).issubset(classes)
    medians = fit_medians(training, np.arange(len(training)), FEATURE_BLOCK)
    np.save(RUN_DIR / "training_medians.npy", medians)
    region_pairs.to_csv(RUN_DIR / "region_pairs.csv", index=False)
    training_item.obs.to_csv(RUN_DIR / "training_cells.csv")
    heldout_item.obs.to_csv(RUN_DIR / "heldout_cells.csv")
    input_signature = hashlib.sha256(json.dumps([
        {"路径": str(p), "大小": p.stat().st_size, "修改时间": p.stat().st_mtime_ns}
        for p in (TRAINING_PATH, TARGET_OBS_PATH)], sort_keys=True).encode()).hexdigest()
    identity = {"官方提交": OFFICIAL_COMMIT, "源码哈希": SOURCE_HASHES, "原始输入签名": input_signature,
        "区域轴哈希": hash_text(feature_var.index), "训练ID哈希": hash_text(training_item.obs_names),
        "目标标签哈希": hash_text(target_obs["Subclass"]), "留出ID哈希": hash_text(heldout_item.obs_names),
        "模式": RUN_MODE, "AE轮数": AE_EPOCHS, "扩散轮数": DIFFUSION_EPOCHS, "批大小": BATCH_SIZE,
        "潜空间维数": 32, "扩散步数": 1000, "随机种子": SEED, "填补": "训练集逐区域中位数",
        "空间条件": None, "图": "仅自环", "输入归一化": None, "输出后处理": None,
        "训练形状": list(training.shape), "生成形状": [len(target_obs), len(feature_var)],
        "依赖": {name: package_metadata.version(name) for name in ("torch", "numpy", "torch-geometric", "diffusers", "huggingface-hub")}}
    source_file = Path(__file__).resolve()
    identity['入口代码SHA256'] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    identity_path = RUN_DIR / "config.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("运行配置或输入变化，禁止复用原检查点")
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"训练形状": training.shape, "生成形状": (len(target_obs), len(feature_var))}, flush=True)


    def save_checkpoint(path,model,optimizer,scheduler,epoch,history):
        partial = Path(str(path)+'.partial')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'history':history,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state(DEVICE)},partial)
        partial.replace(path)

    def train_model(model,epochs,name,batch_loss):
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        checkpoint_path = RUN_DIR/f'{name}.pt'
        start,history = 0,[]
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path,map_location=DEVICE,weights_only=False)
            model.load_state_dict(state['model']); start,history = state['epoch'],state['history']
            if start < epochs:
                optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu(),DEVICE)
            del state
        for epoch in range(start+1,epochs+1):
            model.train(); total,count = 0.,0
            began = time.monotonic()
            for begin in range(0,len(training),BATCH_SIZE):
                rows = np.arange(begin,min(begin+BATCH_SIZE,len(training)))
                optimizer.zero_grad()
                loss = batch_loss(model,rows)
                if not torch.isfinite(loss): raise FloatingPointError(f'{name}损失非有限值')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                optimizer.step(); total += float(loss.detach())*len(rows); count += len(rows)
            scheduler.step()
            record = {'轮次':epoch,'MSE':total/count,'耗时秒':time.monotonic()-began,
                '峰值显存GB':torch.cuda.max_memory_allocated(DEVICE)/1024**3}
            history.append(record)
            pd.DataFrame(history).to_csv(RUN_DIR/f'{name}_history.csv',index=False)
            (RUN_DIR/'status.json').write_text(json.dumps({'阶段':name,**record,'总轮数':epochs},ensure_ascii=False),encoding='utf-8')
            if epoch==1 or epoch%10==0 or epoch==epochs: print(name,record,flush=True)
            if epoch%25==0 or epoch==epochs: save_checkpoint(checkpoint_path,model,optimizer,scheduler,epoch,history)
        del optimizer,scheduler
        gc.collect(); torch.cuda.empty_cache()
        return model

    def ae_loss(model,rows):
        values = torch.from_numpy(fill_rows(training,rows,medians)).to(DEVICE)
        latent,reconstruction = model(values,self_loop_edges(len(rows),DEVICE))
        return F.mse_loss(reconstruction,values)

    def diffusion_loss(model,rows):
        clean = torch.from_numpy(normalized_latent[rows]).to(DEVICE).unsqueeze(1)
        labels = torch.from_numpy(class_indices[rows]).to(DEVICE)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0,1000,(len(rows),),device=DEVICE,dtype=torch.long)
        noisy = noise_scheduler.add_noise(clean,noise,timesteps)
        return F.mse_loss(predict_noise(model,noisy,timesteps,labels),noise)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    autoencoder = SpaAE(input_dim=training.shape[1],gat_dim=[512,32],block_out_dims=[32,32]).to(DEVICE)
    print('原AE参数量',sum(p.numel() for p in autoencoder.parameters()),flush=True)
    autoencoder = train_model(autoencoder,AE_EPOCHS,'autoencoder',ae_loss)
    autoencoder.eval()
    latent_parts = []
    with torch.no_grad():
        for begin in range(0,len(training),BATCH_SIZE):
            rows = np.arange(begin,min(begin+BATCH_SIZE,len(training)))
            values = torch.from_numpy(fill_rows(training,rows,medians)).to(DEVICE)
            latent,_ = autoencoder(values,self_loop_edges(len(rows),DEVICE))
            latent_parts.append(latent.cpu().numpy())
    latent = np.concatenate(latent_parts)
    normalizer = MinMaxNormalize(latent,dim=0)
    if np.any(normalizer.max == normalizer.min): raise ValueError('AE潜变量恒定，停止扩散训练')
    normalized_latent = normalizer.normalize(latent).astype(np.float32)
    class_indices = np.searchsorted(classes,train_labels).astype(np.int64)
    np.savez(RUN_DIR/'latent.npz',latent=latent,normalized=normalized_latent,
        minimum=normalizer.min,maximum=normalizer.max,classes=classes)
    autoencoder.cpu(); gc.collect(); torch.cuda.empty_cache()
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    denoiser = SpaUNet1DModel(in_channels=17,out_channels=1,num_class_embeds=len(classes)).to(DEVICE)
    denoiser = train_model(denoiser,DIFFUSION_EPOCHS,'denoiser',diffusion_loss)

    denoiser.eval(); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    target_classes = torch.from_numpy(np.searchsorted(classes,target_obs['Subclass'].to_numpy()).astype(np.int64)).to(DEVICE)
    generated_latent = torch.randn((len(target_obs),1,32),device=DEVICE)
    with torch.no_grad():
        for timestep in noise_scheduler.timesteps:
            predicted = predict_noise(denoiser,generated_latent,timestep,target_classes)
            generated_latent = noise_scheduler.step(predicted,timestep,generated_latent).prev_sample
    decoded_latent = normalizer.denormalize(generated_latent.squeeze(1).cpu().numpy()).astype(np.float32)
    np.save(RUN_DIR/'generated_latent.npy',decoded_latent)
    denoiser.cpu(); gc.collect(); torch.cuda.empty_cache()
    autoencoder.to(DEVICE)
    generated = autoencoder.decode(torch.from_numpy(decoded_latent).to(DEVICE),
        self_loop_edges(len(target_obs),DEVICE)).cpu().numpy()
    output_path = RUN_DIR/'data.h5ad'
    write_generated(output_path,generated,target_obs,feature_var)
    autoencoder.cpu(); gc.collect(); torch.cuda.empty_cache()
    result = ad.read_h5ad(output_path,backed='r')
    try:
        assert result.shape == generated.shape and result.var_names.equals(feature_var.index)
        assert result.obs_names.equals(target_obs.index)
        np.testing.assert_array_equal(result.obs['Subclass'].astype(str),target_obs['Subclass'])
        for begin in range(0,len(generated),32):
            np.testing.assert_array_equal(result.X[begin:begin+32],generated[begin:begin+32])
    finally: result.file.close()
    diagnostics = {'状态':'完成','模式':RUN_MODE,'正式结果':RUN_MODE=='full','形状':list(generated.shape),
        '缺失比例':float(np.isnan(generated).mean()),'越界比例':float(((generated<0)|(generated>1)).mean()),
        '最小值':float(generated.min()),'最大值':float(generated.max()),
        '唯一生成潜变量数':int(len(np.unique(decoded_latent,axis=0))),
        'AE完成轮数':AE_EPOCHS,'扩散完成轮数':DIFFUSION_EPOCHS,'空间条件':None,
        '输出后处理':None,'官方源码逐文件一致':True,'完整输出逐元素核对':True,
        '训练集全缺失区域数':int(np.isnan(training).all(axis=0).sum())}
    (RUN_DIR/'verification.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(diagnostics,flush=True)
    return output_path


def main(argv=None):
    args = parse_arguments('STADiffuser', argv, spatial_diffusion=True)
    return run(args.training, args.target, args.output, args.seed, args.mode, args.device)


if __name__ == '__main__':
    main()
