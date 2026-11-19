"""内部参照、候选训练和选模诊断。"""
import hashlib, json, math, random
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from data.training import FullMethylationDataset
from model.src.model import LatentDiffusionVAE, latent_diffusion_vae_loss, sample_latent_ddim
from evaluate.A3.metrics import milisi


from model.src.postprocessing import write_json

def fit_selection_reference(matrix, fit_rows, validation_rows, labels,
                            natural_count, top_regions=50000, components=50):
    """只在内部训练细胞的原始区域拟合参照；验证矩阵只做投影。"""
    fit_rows, validation_rows = np.asarray(fit_rows), np.asarray(validation_rows)
    if np.intersect1d(fit_rows, validation_rows).size:
        raise ValueError("内部训练与验证细胞重叠")
    fit_labels = np.asarray(labels)[fit_rows]
    levels = np.unique(fit_labels)
    if len(levels) < 2 or len(levels) >= len(fit_rows):
        raise ValueError("内部训练的 ANOVA 需要至少两类且有重复细胞")
    fill = np.full(natural_count, np.nan)
    scores = np.full(natural_count, np.nan)
    for start in range(0, natural_count, 512):
        columns = np.arange(start, min(start + 512, natural_count))
        values = np.array(matrix[np.ix_(fit_rows, columns)], dtype=np.float64)
        available = np.isfinite(values).any(0)
        medians = np.full(len(columns), np.nan)
        medians[available] = np.nanmedian(values[:, available], axis=0)
        fill[columns] = medians
        values = values[:, available]
        np.copyto(values, medians[available][None, :], where=~np.isfinite(values))
        grand = values.mean(0)
        between, within = np.zeros(values.shape[1]), np.zeros(values.shape[1])
        for level in levels:
            group = values[fit_labels == level]
            mean = group.mean(0)
            between += len(group) * (mean - grand) ** 2
            within += ((group - mean) ** 2).sum(0)
        numerator, denominator = between / (len(levels) - 1), within / (len(fit_rows) - len(levels))
        local = np.divide(numerator, denominator, out=np.full(values.shape[1], np.nan), where=denominator > 0)
        local[(denominator == 0) & (numerator > 0)] = np.inf
        scores[columns[available]] = local
    eligible = np.flatnonzero(np.isfinite(scores) | np.isposinf(scores))
    if len(eligible) < 2:
        raise ValueError("内部训练没有足够的可评分原始区域")
    selected = np.sort(eligible[np.argsort(scores[eligible], kind="stable")[-min(top_regions, len(eligible)):]])
    fill[~np.isfinite(fill)] = np.median(fill[np.isfinite(fill)])
    def prepared(rows):
        values = np.array(matrix[np.ix_(rows, selected)], dtype=np.float32)
        np.copyto(values, fill[selected][None, :], where=~np.isfinite(values))
        return values
    pca = PCA(n_components=min(components, len(fit_rows)-1, len(selected)),
              svd_solver="randomized", random_state=42)
    pca.fit(prepared(fit_rows))
    return {"selected": selected, "fill": fill, "pca": pca,
            "validation_rep": pca.transform(prepared(validation_rows)),
            "validation_labels": np.asarray(labels)[validation_rows],
            "fit_cells": len(fit_rows), "validation_cells": len(validation_rows)}


def selection_rank(record):
    """通过原验收的候选优先；只用内部验证 miLISI 排名。"""
    score = float(record["selection_milisi"])
    return (not bool(record["selection_eligible"]) or not np.isfinite(score),
            -score if np.isfinite(score) else float("inf"), float(record["selection_score"]))


def vae_state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("denoiser."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_neighborhoods(real_rep, generated_rep, k=30):
    """与正式指标同一邻域，额外报告两侧混合，防止单侧混合被汇总掩盖。"""
    combined = np.vstack([real_rep, generated_rep])
    raw = NearestNeighbors(n_neighbors=k+1).fit(combined).kneighbors(combined, return_distance=False)
    indices = np.array([row[row != i][:k] for i, row in enumerate(raw)])
    p_real = (indices < len(real_rep)).mean(1)
    return {"真实侧异来源近邻比例": float((1-p_real[:len(real_rep)]).mean()),
            "合成侧异来源近邻比例": float(p_real[len(real_rep):].mean()),
            "合成侧全同源邻域比例": float((p_real[len(real_rep):] == 0).mean())}

def train_candidate(matrix, retained_rows, medians, labels, class_names,
                    train_positions, validation_positions, beta_target,
                    candidate_dir, device, checkpoint_metadata, config, *, selection_reference, frozen_vae=True):
    # 以局部配置绑定训练常量；不同运行之间不共享可变全局配置。
    DEVICE = device
    SEED = config.SEED
    LATENT_DIM = config.LATENT_DIM
    CONDITION_DIM = config.CONDITION_DIM
    HIDDEN_DIM = config.HIDDEN_DIM
    DENOISER_BLOCKS = config.DENOISER_BLOCKS
    DIFFUSION_TIMESTEPS = config.DIFFUSION_TIMESTEPS
    DDIM_SAMPLING_STEPS = config.DDIM_SAMPLING_STEPS
    DIFFUSION_LOSS_WEIGHT = config.DIFFUSION_LOSS_WEIGHT
    DIFFUSION_PREDICTION = config.DIFFUSION_PREDICTION
    DIFFUSION_SCHEDULE = config.DIFFUSION_SCHEDULE
    DROPOUT_RATE = config.DROPOUT_RATE
    FEATURE_BLOCK = config.FEATURE_BLOCK
    TRAIN_BATCH_SIZE = config.TRAIN_BATCH_SIZE
    VAE_PRETRAIN_EPOCHS = config.VAE_PRETRAIN_EPOCHS
    DIFFUSION_WARMUP_EPOCHS = config.DIFFUSION_WARMUP_EPOCHS
    JOINT_EPOCHS = config.JOINT_EPOCHS
    JOINT_VAE_LR_FACTOR = config.JOINT_VAE_LR_FACTOR
    SELECTION_INTERVAL = config.SELECTION_INTERVAL
    SELECTION_FEATURES = config.SELECTION_FEATURES
    DIFFUSION_PATIENCE = config.DIFFUSION_PATIENCE
    PATIENCE = config.PATIENCE
    EARLY_STOP_MIN_DELTA = config.EARLY_STOP_MIN_DELTA
    LEARNING_RATE = config.LEARNING_RATE
    KL_WARMUP_EPOCHS = config.KL_WARMUP_EPOCHS

    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def _make_loaders(
        matrix: np.memmap,
        retained_rows: np.ndarray,
        medians: np.ndarray,
        labels: np.ndarray,
        train_positions: np.ndarray,
        validation_positions: np.ndarray,
    ) -> tuple[DataLoader, DataLoader]:
        generator = torch.Generator().manual_seed(SEED)
        train_dataset = FullMethylationDataset(
            matrix, retained_rows, train_positions, medians, labels,
            device=DEVICE if DEVICE.type == "cuda" else None,
        )
        validation_dataset = FullMethylationDataset(
            matrix, retained_rows, validation_positions, medians, labels,
            device=DEVICE if DEVICE.type == "cuda" else None,
        )
        common = {"batch_size": TRAIN_BATCH_SIZE, "num_workers": 0, "pin_memory": False}
        return (
            DataLoader(train_dataset, shuffle=True, generator=generator, **common),
            DataLoader(validation_dataset, shuffle=False, **common),
        )


    def _run_epoch(
        model, loader, optimizer, beta, device, random_seed=None,
        stage="joint", diffusion_weight=DIFFUSION_LOSS_WEIGHT,
    ):
        training = optimizer is not None
        model.train(training)
        sums = np.zeros(4, dtype=np.float64)
        observed_count = cell_count = 0
        generator = None if random_seed is None else torch.Generator(device=device).manual_seed(random_seed)
        with torch.set_grad_enabled(training):
            for batch, observed, classes in loader:
                batch, observed, classes = [value.to(device) for value in (batch, observed, classes)]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                mu, logvar = model.encode(batch)
                noise = torch.randn(mu.shape, device=device, generator=generator)
                z = model.reparameterize(mu, logvar, noise)
                reconstruction = model.decode(z)
                z0 = model.normalize_latent(z)
                if stage == "vae_pretrain":
                    prediction = target = torch.zeros_like(z0)
                else:
                    times = torch.randint(model.diffusion_timesteps, (len(batch),), device=device, generator=generator)
                    epsilon = torch.randn(z0.shape, device=device, generator=generator)
                    prediction = model.predict_velocity(model.add_noise(z0, times, epsilon), times, classes)
                    target = model.velocity_target(z0, times, epsilon)
                loss, recon, kl, diffusion = latent_diffusion_vae_loss(
                    reconstruction, batch, observed, mu, logvar, prediction, target,
                    beta, 0.0 if stage == "vae_pretrain" else diffusion_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("训练损失出现非有限值")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                count = int(observed.sum())
                sums += np.array([recon.item() * count, kl.item() * len(batch), diffusion.item() * len(batch), 0.0])
                observed_count += count
                cell_count += len(batch)
        recon, kl, diffusion = sums[:3] / np.array([observed_count, cell_count, cell_count])
        weight = 0.0 if stage == "vae_pretrain" else diffusion_weight
        return {"loss": float(recon + beta * kl + weight * diffusion), "reconstruction": float(recon), "kl": float(kl), "diffusion": float(diffusion)}


    def fit_model_statistics(matrix, rows, medians):
        # 只读取内部训练子集；验证细胞不参与填补、中心化或评估特征选择。
        center = np.empty(matrix.shape[1], dtype=np.float32)
        variance = np.empty_like(center)
        for start in range(0, matrix.shape[1], FEATURE_BLOCK):
            stop = min(start + FEATURE_BLOCK, matrix.shape[1])
            values = np.asarray(matrix[np.ix_(rows, np.arange(start, stop))], dtype=np.float32)
            mask = np.isfinite(values)
            count = mask.sum(0)
            values = np.where(mask, values, 0.0)
            mean = values.sum(0, dtype=np.float64) / np.maximum(count, 1)
            center[start:stop] = np.where(count, mean, medians[start:stop])
            variance[start:stop] = np.maximum(np.square(values, dtype=np.float64).sum(0) / np.maximum(count, 1) - mean ** 2, 0)
        features = np.sort(np.argsort(variance)[-min(SELECTION_FEATURES, len(variance)):])
        return center, features


    @torch.no_grad()
    def cache_posterior(model, loader, device):
        model.eval()
        means, logvars, labels = [], [], []
        for batch, _, classes in loader:
            mu, logvar = model.encode(batch.to(device))
            means.append(mu.detach())
            logvars.append(logvar.detach())
            labels.append(classes.to(device))
        return torch.cat(means), torch.cat(logvars), torch.cat(labels)


    def diffusion_cached_epoch(model, cached, optimizer, device, seed=None):
        training = optimizer is not None
        model.eval()
        model.denoiser.train(training)
        mu, logvar, labels = cached
        generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        order = torch.randperm(len(mu), device=device, generator=generator) if training else torch.arange(len(mu), device=device)
        total = 0.0
        with torch.set_grad_enabled(training):
            for indices in order.split(TRAIN_BATCH_SIZE):
                posterior_noise = torch.randn(mu[indices].shape, device=device, generator=generator)
                z0 = model.normalize_latent(model.reparameterize(mu[indices], logvar[indices], posterior_noise).detach())
                times = torch.randint(model.diffusion_timesteps, (len(indices),), device=device, generator=generator)
                epsilon = torch.randn(z0.shape, device=device, generator=generator)
                prediction = model.predict_velocity(model.add_noise(z0, times, epsilon), times, labels[indices])
                loss = (prediction - model.velocity_target(z0, times, epsilon)).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("扩散预热损失出现非有限值")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 5.0)
                    optimizer.step()
                total += loss.item() * len(indices)
        return total / len(mu)


    @torch.no_grad()
    def evaluate_representation(model, loader, device):
        model.eval()
        errors = np.zeros(3, dtype=np.float64)
        count = 0
        means = []
        for batch, observed, _ in loader:
            batch, observed = batch.to(device), observed.to(device)
            mu, _ = model.encode(batch)
            for index, output in enumerate((model.decode(mu), model.decode(mu.roll(1, 0)), model.input_center.expand_as(batch))):
                errors[index] += (output - batch).square().masked_select(observed).sum().item()
            count += int(observed.sum())
            means.append(mu.cpu())
        recon, shuffled, baseline = errors / count
        latent_variance = torch.cat(means).var(dim=0, unbiased=False)
        return {
            "mean_reconstruction": float(recon), "shuffle_reconstruction": float(shuffled),
            "baseline_reconstruction": float(baseline),
            "reconstruction_gain": float(1 - recon / max(baseline, 1e-12)),
            "shuffle_gain": float((shuffled - recon) / max(baseline, 1e-12)),
            "active_units": int((latent_variance > 1e-3).sum()),
        }


    @torch.no_grad()
    def evaluate_selection_score(model, loader, device, features):
        model.eval()
        validation = _run_epoch(model, loader, None, 0.0, device, SEED + 20_000)
        representation = evaluate_representation(model, loader, device)
        real, observed, labels = [], [], []
        for batch, mask, classes in loader:
            real.append(batch[:, features].cpu().numpy())
            observed.append(mask[:, features].cpu().numpy())
            labels.append(classes.cpu().numpy())
        real, observed, labels = np.concatenate(real), np.concatenate(observed), np.concatenate(labels)

        def generate(classes):
            pieces = []
            for start in range(0, len(classes), 64):
                condition = torch.as_tensor(classes[start:start + 64], device=device)
                latent = sample_latent_ddim(model, condition, DDIM_SAMPLING_STEPS, SEED + 30_000 + start)
                pieces.append(model.decode(latent)[:, output_features].cpu().numpy())
            return np.concatenate(pieces)

        scoring_features = np.asarray(selection_reference["selected"])
        output_features = np.union1d(features, scoring_features)
        old_positions = np.searchsorted(output_features, features)
        scoring_positions = np.searchsorted(output_features, scoring_features)
        # 生成同一批完整解码结果中的所需列，不改网络、不改采样器。
        generated_all = generate(labels)
        generated = generated_all[:, old_positions]
        generated_rep = selection_reference["pca"].transform(generated_all[:, scoring_positions])
        # 样本不足时保持局部邻域，避免把整个内部验证集合纳入每个细胞的近邻。
        selection_neighbors = 30 if 2 * len(labels) > 30 else max(1, len(labels) - 1)
        validation_milisi, _, _ = milisi(selection_reference["validation_rep"], labels,
                                         generated_rep, labels, n_neighbors=selection_neighbors)
        permuted_labels = np.random.default_rng(SEED).permutation(labels)
        permuted = generate(permuted_labels)[:, old_positions]
        counts = observed.sum(0)
        mean = (real * observed).sum(0) / np.maximum(counts, 1)
        variance = (((real - mean) ** 2) * observed).sum(0) / np.maximum(counts, 1)
        valid = (counts >= 10) & (variance > 1e-4)
        if not valid.any():
            raise ValueError("验证集中没有足够的可观测变异特征")
        scale = np.maximum(variance[valid], 1e-4)
        generated_variance = generated[:, valid].var(0)
        mean_error = float(np.mean((generated[:, valid].mean(0) - mean[valid]) ** 2 / scale))
        variance_error = float(np.mean(np.abs(np.log((generated_variance + 1e-6) / (scale + 1e-6)))))
        diversity_ratio = float(np.median(np.sqrt(generated_variance / scale)))

        def centroid_error(samples):
            errors = []
            for label in np.unique(labels):
                rows = labels == label
                if rows.sum() < 3:
                    continue
                mask = observed[rows][:, valid]
                count = mask.sum(0)
                keep = count >= 2
                if not keep.any():
                    continue
                reference = (real[rows][:, valid] * mask).sum(0) / np.maximum(count, 1)
                errors.append(np.mean(((samples[rows][:, valid].mean(0) - reference) ** 2 / scale)[keep]))
            return float(np.mean(errors)) if errors else 0.0

        conditional_error = centroid_error(generated)
        eligible = bool(representation["reconstruction_gain"] > 0.005 and representation["shuffle_gain"] > 0.005 and representation["active_units"] >= 4 and diversity_ratio > 0.1)
        return {
            **representation,
            "selection_milisi": float(validation_milisi),
            "selection_neighbors": selection_neighbors,
            **source_neighborhoods(selection_reference["validation_rep"], generated_rep, k=selection_neighbors),
            "selection_reconstruction": validation["reconstruction"],
            "selection_kl": validation["kl"], "selection_diffusion": validation["diffusion"],
            "selection_mean_error": mean_error, "selection_variance_error": variance_error,
            "selection_centroid_error": conditional_error, "selection_diversity_ratio": diversity_ratio,
            "condition_permuted_centroid_error": centroid_error(permuted),
            "condition_output_rmse": float(np.sqrt(np.mean((generated - permuted) ** 2))),
            "selection_eligible": eligible,
            "selection_score": float(validation["reconstruction"] / max(representation["baseline_reconstruction"], 1e-12) + mean_error + variance_error + conditional_error),
        }


    def train_cvae_candidate(matrix, retained_rows, medians, labels, class_names,
                             train_positions, validation_positions, beta_target,
                             candidate_dir, device, checkpoint_metadata):
        _seed_everything(SEED)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        train_loader, validation_loader = _make_loaders(matrix, retained_rows, medians, labels, train_positions, validation_positions)
        model = LatentDiffusionVAE(matrix.shape[1], len(class_names), LATENT_DIM, CONDITION_DIM,
                                  HIDDEN_DIM, DROPOUT_RATE, DIFFUSION_TIMESTEPS, DENOISER_BLOCKS).to(device)
        center, features = fit_model_statistics(matrix, retained_rows[train_positions], medians)
        model.input_center.copy_(torch.as_tensor(center, device=device))
        # 用训练集均值初始化输出截距；细胞残差仍必须由潜变量学习。
        with torch.no_grad():
            model.decoder[-2].bias.copy_(torch.logit(model.input_center.clamp(1e-4, 1 - 1e-4)))
        vae_parameters = [p for name, p in model.named_parameters() if not name.startswith("denoiser.")]
        history = []
        selection_history = []
        history_path = candidate_dir / "training_history.csv"
        best_epoch = 0

        def snapshot():
            return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        def log(stage, local_epoch, train, validation, extra=None):
            record = {"epoch": len(history) + 1, "stage": stage, "stage_epoch": local_epoch,
                      "beta_target": beta_target, **{f"train_{k}": v for k, v in train.items()},
                      **{f"validation_{k}": v for k, v in validation.items()}, **(extra or {})}
            history.append(record)
            pd.DataFrame(history).to_csv(history_path, index=False)
            print(f"β={beta_target:g} 阶段={stage} 轮次={local_epoch} 验证={validation.get('loss', float('nan')):.6f} "
                  f"重构={validation.get('reconstruction', float('nan')):.6f} 扩散={validation.get('diffusion', float('nan')):.6f}", flush=True)

        optimizer = torch.optim.Adam(vae_parameters, lr=LEARNING_RATE)
        best_loss, stale, best_state = math.inf, 0, snapshot()
        for epoch in range(1, VAE_PRETRAIN_EPOCHS + 1):
            beta = beta_target * min(1.0, max(0, epoch - 5) / KL_WARMUP_EPOCHS)
            train = _run_epoch(model, train_loader, optimizer, beta, device, stage="vae_pretrain")
            val = _run_epoch(model, validation_loader, None, beta, device, SEED + 10_000, stage="vae_pretrain")
            log("vae_pretrain", epoch, train, val)
            if epoch < 5 + KL_WARMUP_EPOCHS:
                continue
            if val["reconstruction"] < best_loss - EARLY_STOP_MIN_DELTA:
                best_loss, stale, best_state = val["reconstruction"], 0, snapshot()
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        model.load_state_dict(best_state)
        del optimizer, best_state
        representation = evaluate_representation(model, validation_loader, device)
        print("VAE 预训练检查", json.dumps(representation, ensure_ascii=False), flush=True)
        (candidate_dir / "pretrain_diagnostics.json").write_text(json.dumps(representation, indent=2), encoding="utf-8")

        cached_train = cache_posterior(model, train_loader, device)
        cached_validation = cache_posterior(model, validation_loader, device)
        with torch.no_grad():
            mu, logvar, _ = cached_train
            model.latent_center.copy_(mu.mean(0))
            model.latent_scale.copy_((mu.var(0, unbiased=False) + logvar.exp().mean(0)).sqrt().clamp_min(1e-3))
        for parameter in vae_parameters:
            parameter.requires_grad_(False)
        vae_before = vae_state_hash(model)
        optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=LEARNING_RATE)
        best_denoiser, best_selection, best_rank = None, None, (True, math.inf, math.inf)
        last_improvement = 0
        best_diffusion_epoch = 0
        for epoch in range(1, DIFFUSION_WARMUP_EPOCHS + 1):
            train = diffusion_cached_epoch(model, cached_train, optimizer, device)
            val = diffusion_cached_epoch(model, cached_validation, None, device, SEED + 10_000)
            if epoch == 1 or epoch % 10 == 0:
                log("diffusion_warmup", epoch, {"loss": train, "diffusion": train}, {"loss": val, "diffusion": val})
            if epoch == 1 or epoch % SELECTION_INTERVAL == 0 or epoch == DIFFUSION_WARMUP_EPOCHS:
                selection = evaluate_selection_score(model, validation_loader, device, features)
                selection_history.append({"stage": "diffusion_warmup", "stage_epoch": epoch, **selection})
                pd.DataFrame(selection_history).to_csv(candidate_dir / "selection_history.csv", index=False)
                rank = selection_rank(selection)
                print(f"扩散轮次 {epoch}；内部验证 miLISI={selection['selection_milisi']:.6f}；通过验收={selection['selection_eligible']}", flush=True)
                if best_denoiser is None or rank < best_rank:
                    best_rank, best_selection, last_improvement = rank, selection, epoch
                    best_diffusion_epoch = epoch
                    best_denoiser = {k: v.detach().cpu().clone() for k, v in model.denoiser.state_dict().items()}
            if epoch >= 100 and epoch - last_improvement >= DIFFUSION_PATIENCE:
                break
        diffusion_epochs_completed = epoch
        model.denoiser.load_state_dict(best_denoiser)
        del cached_train, cached_validation, optimizer, best_denoiser
        if frozen_vae:
            best_state = snapshot()
            best_epoch = best_diffusion_epoch
        else:
            for parameter in vae_parameters:
                parameter.requires_grad_(True)

            optimizer = torch.optim.Adam([
                {"params": vae_parameters, "lr": LEARNING_RATE * JOINT_VAE_LR_FACTOR},
                {"params": model.denoiser.parameters(), "lr": LEARNING_RATE},
            ])
            best_state, best_selection, stale = None, None, 0
            for epoch in range(1, JOINT_EPOCHS + 1):
                weight = DIFFUSION_LOSS_WEIGHT * min(1.0, epoch / 10)
                train = _run_epoch(model, train_loader, optimizer, beta_target, device, diffusion_weight=weight)
                val = _run_epoch(model, validation_loader, None, beta_target, device, SEED + 10_000, diffusion_weight=weight)
                log("joint", epoch, train, val, {"diffusion_weight": weight})
                if epoch % SELECTION_INTERVAL and epoch != JOINT_EPOCHS:
                    continue
                selection = evaluate_selection_score(model, validation_loader, device, features)
                selection_history.append({"stage": "joint", "stage_epoch": epoch, **selection})
                pd.DataFrame(selection_history).to_csv(candidate_dir / "selection_history.csv", index=False)
                print("内部验证诊断", json.dumps(selection, ensure_ascii=False), flush=True)
                # 先比较是否通过塌缩检查，再比较生成质量，绝不按原始 KL 排名。
                rank = selection_rank(selection)
                old_rank = selection_rank(best_selection) if best_selection else (True, math.inf, math.inf)
                if best_state is None or rank < old_rank:
                    best_state, best_selection, stale = snapshot(), selection, 0
                    best_epoch = len(history)
                else:
                    stale += 1
                if epoch >= 20 and stale >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        selection = evaluate_selection_score(model, validation_loader, device, features)
        vae_after = vae_state_hash(model)
        if frozen_vae and vae_before != vae_after:
            raise AssertionError("冻结扩散训练修改了 VAE 参数或统计量")
        model_path = candidate_dir / "model.pt"
        checkpoint = {
            "state_dict": best_state, "input_dim": int(matrix.shape[1]), "latent_dim": LATENT_DIM,
            "condition_dim": CONDITION_DIM, "hidden_dim": HIDDEN_DIM, "dropout_rate": DROPOUT_RATE,
            "denoiser_blocks": DENOISER_BLOCKS, "diffusion_timesteps": DIFFUSION_TIMESTEPS,
            "sampling_steps": DDIM_SAMPLING_STEPS, "diffusion_schedule": DIFFUSION_SCHEDULE,
            "diffusion_prediction": DIFFUSION_PREDICTION, "diffusion_loss_weight": DIFFUSION_LOSS_WEIGHT,
            "training_stages": ["vae_pretrain", "diffusion_warmup"] + ([] if frozen_vae else ["joint"]),
            "frozen_vae": frozen_vae, "selection_metric": f"内部验证真实标签 {selection['selection_neighbors']}-NN miLISI",
            "vae_state_before_diffusion": vae_before, "vae_state_after_training": vae_after,
            "latent_normalization": "预训练后由内部训练后验拟合并固定",
            "class_names": class_names.tolist(), "seed": SEED, "beta_target": beta_target,
            "epochs_completed": int(sum(r["stage"] != "diffusion_warmup" for r in history) + diffusion_epochs_completed),
            "best_epoch": best_epoch, "best_validation_loss": selection["selection_reconstruction"],
            **selection, **checkpoint_metadata,
        }
        partial = model_path.with_suffix(".pt.partial")
        torch.save(checkpoint, partial)
        partial.replace(model_path)
        result = {k: v for k, v in checkpoint.items() if k not in {"state_dict", "class_names"}}
        result.update(candidate_dir=str(candidate_dir), model_path=str(model_path), history_path=str(history_path))
        write_json(candidate_dir / "diagnostics.json", result)
        del model, best_state, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    return train_cvae_candidate(matrix, retained_rows, medians, labels, class_names,
                                train_positions, validation_positions, beta_target,
                                candidate_dir, device, checkpoint_metadata)
