"""
MOGCRL训练脚本（TRPO-QP 专用）

使用TRPO-QP约束策略进行训练，配置文件中算法字段已固定。
"""

import os
import sys
import json
import copy
import argparse
import time
import functools
import queue
import yaml
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from contextlib import nullcontext
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mogcrl.nets.policy import CategoricalPolicy
from mogcrl.nets.critic import MultiHeadCritic
from typing import Dict, Any, Optional, List, Tuple
from types import SimpleNamespace

from mogcrl.memory import PolicyBufferStore, OnPolicyBuffer
from mogcrl.runners.vector_runner import VectorRunner
from mogcrl.runners.async_workers import start_workers
from mogcrl.core.update import trpo_qp_update_actor, update_critic
from mogcrl.utils.metrics import aggregate_env_metrics
from mogcrl.utils.seeding import set_seed
from mogcrl.utils.logger import ExpLogger
from mogcrl.nets.encoders import FlatEncoder18D, PaperObsEncoder
from mogcrl.irdc.irdc import IRDC
from mogcrl.agents import AgentRouting, build_registry
from mogcrl.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    cleanup_checkpoints,
    save_best_meta,
)
from metrics import export_csv, export_tb
from envs import MockWarehouseAdapter


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def _deep_update(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> None:
    if not src:
        return
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def _resolve_amp_flags(
    training_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[bool, bool, bool, torch.dtype, str]:
    amp_cfg = training_cfg.get('amp', False)
    if isinstance(amp_cfg, bool):
        amp_enabled_cfg = bool(amp_cfg)
        amp_policy_cfg = bool(amp_cfg)
        amp_critic_cfg = bool(amp_cfg)
        amp_dtype_cfg = 'float16'
    else:
        amp_enabled_cfg = bool(amp_cfg.get('enabled', False))
        amp_policy_cfg = bool(amp_cfg.get('policy', amp_enabled_cfg))
        amp_critic_cfg = bool(amp_cfg.get('critic', amp_enabled_cfg))
        amp_dtype_cfg = str(amp_cfg.get('dtype', 'float16')).lower()

    device_type = device.type
    amp_supported = (device_type == 'cuda' and torch.cuda.is_available())

    if amp_dtype_cfg in ('bf16', 'bfloat16'):
        amp_dtype = torch.bfloat16
        amp_dtype_cfg = 'bfloat16'
    else:
        amp_dtype = torch.float16
        amp_dtype_cfg = 'float16'

    amp_enabled = amp_supported and amp_enabled_cfg
    critic_amp_enabled = amp_enabled and amp_critic_cfg
    policy_amp_requested = amp_enabled and amp_policy_cfg

    return amp_enabled, critic_amp_enabled, policy_amp_requested, amp_dtype, amp_dtype_cfg


def build_policy_module(
    device: torch.device,
    obs_dim: int,
    act_dim: int,
    hidden: Optional[list],
    use_rnn: bool,
    rnn_cfg: Dict[str, Any],
    use_gat: bool,
    gat_cfg: Dict[str, Any],
    gat_fuse: str,
) -> CategoricalPolicy:
    module = CategoricalPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=hidden or [128, 128],
        use_rnn=use_rnn,
        rnn_cfg=rnn_cfg,
        use_gat=use_gat,
        gat_cfg=gat_cfg,
        gat_fuse=gat_fuse,
    )
    return module.to(device)


def build_critic_module(
    device: torch.device,
    obs_dim: int,
    hidden: Optional[list],
    agent_embed_dim: int,
    use_rnn: bool,
    rnn_cfg: Dict[str, Any],
    use_gat: bool,
    gat_cfg: Dict[str, Any],
    gat_fuse: str,
) -> MultiHeadCritic:
    module = MultiHeadCritic(
        obs_dim=obs_dim,
        hidden=hidden or [128, 128],
        agent_embed_dim=agent_embed_dim,
        use_rnn=use_rnn,
        rnn_cfg=rnn_cfg,
        use_gat=use_gat,
        gat_cfg=gat_cfg,
        gat_fuse=gat_fuse,
    )
    return module.to(device)


def record_worker_health(
    iter_metrics: Dict[str, Any],
    ack_versions: Dict[int, int],
    broadcast_ver: int,
    heartbeat_ts: Dict[int, float],
    now_ts: float,
    queue_timeout: float,
) -> None:
    """
    根据 ACK/心跳状态记录告警与延迟
    """
    ack_warn = 0
    heartbeat_warn = 0
    threshold = max(queue_timeout * 10, 5.0)
    for wid, ack_ver in ack_versions.items():
        if ack_ver < broadcast_ver - 1:
            ack_warn += 1
        lag = now_ts - heartbeat_ts.get(wid, 0.0)
        iter_metrics[f'worker{wid}/heartbeat_lag_s'] = lag
        if lag > threshold:
            heartbeat_warn += 1
    if ack_warn:
        iter_metrics['warnings/ack_behind'] = ack_warn
    if heartbeat_warn:
        iter_metrics['warnings/heartbeat_stale'] = heartbeat_warn


def _is_backbone(name: str) -> bool:
    key = name.lower()
    tokens = ["encoder", "rnn", "gat", "backbone", "gru", "lstm"]
    return any(tok in key for tok in tokens)


def _filter_state_dict(module: torch.nn.Module, only_backbone: bool) -> Dict[str, torch.Tensor]:
    state = module.state_dict()
    if not only_backbone:
        return state
    return {k: v for k, v in state.items() if _is_backbone(k)}


def build_backend_adapter(args, config: Dict[str, Any]) -> Any:
    reward_mode = str(config.get('reward_mode', 'engineering_shaping'))
    if args.backend == 'stub':
        return None
    if args.backend == 'mock':
        return MockWarehouseAdapter(
            num_agents=args.num_agents,
            action_dim=config.get('action_dim', 5),
            max_steps=200,
            reward_mode=reward_mode,
        )
    if args.backend == 'fulfillment':
        from envs import FulfillmentAdapter

        return FulfillmentAdapter(
            seed=args.seed,
            max_agents=args.num_agents,
            obs_mode='dict',
            reward_mode=reward_mode,
        )
    if args.backend == 'rware':
        from envs import RWAREAdapter

        return RWAREAdapter(
            seed=args.seed,
            max_agents=args.num_agents,
            obs_mode='dict',
            reward_mode=reward_mode,
        )
    raise ValueError(f"未知 backend: {args.backend}")


def run_evaluation(
    policy_map: Dict[str, torch.nn.Module],
    critic_map: Dict[str, torch.nn.Module],
    routing: AgentRouting,
    eval_cfg: Dict[str, Any],
    args,
    config: Dict[str, Any],
    obs_encoder,
    intrinsic_module,
    frontends_cfg,
    gat_cfg,
    graphs_cfg,
    device_str: str,
    steps_per_agent: int,
    max_steps_per_ep: int,
    primary_policy_key: str,
    primary_critic_key: str,
    reward_source: str,
) -> Dict[str, float]:
    episodes = int(eval_cfg.get('episodes', 1))
    eval_seed = int(eval_cfg.get('seed', 123))
    set_seed(eval_seed)
    adapter = build_backend_adapter(args, config)
    constraints_cfg = config.get('constraints', {}) or {}
    reward_mode = str(config.get('reward_mode', 'engineering_shaping'))
    intrinsic_state = copy.deepcopy(intrinsic_module.state_dict()) if intrinsic_module is not None else None
    eval_runner = VectorRunner(
        backend_adapter=adapter,
        policy=policy_map[primary_policy_key],
        critic=critic_map[primary_critic_key],
        policy_map=policy_map,
        critic_map=critic_map,
        routing=routing,
        buffer_store=None,
        intrinsic_module=intrinsic_module,
        steps_per_agent=steps_per_agent,
        num_agents=args.num_agents,
        obs_dim=obs_encoder.out_dim,
        act_dim=config.get('action_dim', 5),
        episodes=1,
        max_steps_per_ep=max_steps_per_ep,
        device=device_str,
        obs_encoder=obs_encoder,
        obs_config={**(config.get('obs') or {})},
        intrinsic_cfg=config.get('intrinsic', {}),
        frontends_cfg=frontends_cfg,
        gat_cfg=gat_cfg,
        graphs_cfg=graphs_cfg,
        constraints_cfg=constraints_cfg,
        reward_mode=reward_mode,
        reward_source=reward_source,
    )
    policy_modes = {k: m.training for k, m in policy_map.items()}
    critic_modes = {k: m.training for k, m in critic_map.items()}
    for module in policy_map.values():
        module.eval()
    for module in critic_map.values():
        module.eval()

    returns_R = []
    returns_time = []
    returns_batt = []
    viol_time = []
    viol_batt = []
    success_rates = []
    samples = []

    with torch.no_grad():
        for ep in range(episodes):
            set_seed(eval_seed + ep)
            rollout, info = eval_runner.collect()
            rew_dict = rollout.get('rew_dict', {})
            if isinstance(rew_dict, dict):
                if isinstance(rew_dict.get('R'), torch.Tensor):
                    returns_R.append(float(rew_dict['R'].sum().item()))
                if isinstance(rew_dict.get('time'), torch.Tensor):
                    returns_time.append(float(rew_dict['time'].sum().item()))
                if isinstance(rew_dict.get('batt'), torch.Tensor):
                    returns_batt.append(float(rew_dict['batt'].sum().item()))
            viol_time.append(float(info.get('deadline_violation_rate', 0.0)))
            viol_batt.append(float(info.get('battery_violation_rate', 0.0)))
            success_rates.append(float(info.get('success_rate', info.get('completed_tasks', 0.0))))
            samples.append(float(info.get('samples_collected', 0.0)))

    for key, mode in policy_modes.items():
        policy_map[key].train(mode)
    for key, mode in critic_modes.items():
        critic_map[key].train(mode)

    if intrinsic_state is not None and intrinsic_module is not None:
        intrinsic_module.load_state_dict(intrinsic_state)

    def _mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    def _std(values, mean):
        if not values:
            return 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return float(var ** 0.5)

    ret_R_mean = _mean(returns_R)
    ret_R_std = _std(returns_R, ret_R_mean)
    ret_time_mean = _mean(returns_time)
    ret_batt_mean = _mean(returns_batt)
    viol_time_mean = _mean(viol_time)
    viol_batt_mean = _mean(viol_batt)
    sr_mean = _mean(success_rates)
    samples_mean = _mean(samples)

    return {
        'eval/return_R_mean': ret_R_mean,
        'eval/return_R_std': ret_R_std,
        'eval/return_time_mean': ret_time_mean,
        'eval/return_batt_mean': ret_batt_mean,
        'eval/viol_time_mean': viol_time_mean,
        'eval/viol_batt_mean': viol_batt_mean,
        'eval/success_rate': sr_mean,
        'eval/samples_collected': samples_mean,
    }


def make_worker_runner(
    worker_cfg,
    routing: AgentRouting,
    config: Dict[str, Any],
    policy_state: Dict[str, Dict[str, torch.Tensor]],
    critic_state: Dict[str, Dict[str, torch.Tensor]],
    policy_args: Dict[str, Any],
    critic_args: Dict[str, Any],
    obs_cfg: Dict[str, Any],
    frontends_cfg: Dict[str, Any],
    gat_cfg: Dict[str, Any],
    graphs_cfg: Dict[str, Any],
    intrinsic_cfg: Dict[str, Any],
    steps_per_agent: int,
    max_steps_per_ep: int,
    logging_cfg: Dict[str, Any],
    cli_args,
) -> VectorRunner:
    device = torch.device(worker_cfg.device)
    constraints_cfg = config.get('constraints', {}) or {}
    reward_mode = str(config.get('reward_mode', 'engineering_shaping'))
    policy_map = {
        key: build_policy_module(device=device, **policy_args)
        for key in policy_state.keys()
    }
    for key, state in policy_state.items():
        policy_map[key].load_state_dict(state)

    critic_map = {
        key: build_critic_module(device=device, **critic_args)
        for key in critic_state.keys()
    }
    for key, state in critic_state.items():
        critic_map[key].load_state_dict(state)

    use_paper_obs = bool(obs_cfg.get('use_paper_obs', False))
    fov_size = int(obs_cfg.get('fov_size', 3))
    obs_encoder = PaperObsEncoder(fov_size=fov_size) if use_paper_obs else FlatEncoder18D()

    primary_policy_key = next(iter(policy_map.keys()))
    primary_critic_key = next(iter(critic_map.keys()))
    num_agents = max(routing.policy_of.keys()) + 1 if routing.policy_of else steps_per_agent

    adapter_args = SimpleNamespace(
        backend=getattr(cli_args, 'backend', config.get('backend', 'mock')),
        num_agents=num_agents,
        seed=getattr(worker_cfg, 'seed', getattr(cli_args, 'seed', 0)),
    )
    backend_adapter = build_backend_adapter(adapter_args, config)

    runner = VectorRunner(
        backend_adapter=backend_adapter,
        policy=policy_map[primary_policy_key],
        critic=critic_map[primary_critic_key],
        policy_map=policy_map,
        critic_map=critic_map,
        routing=routing,
        buffer_store=None,
        intrinsic_module=None,
        steps_per_agent=steps_per_agent,
        num_agents=num_agents,
        obs_dim=policy_args['obs_dim'],
        act_dim=policy_args['act_dim'],
        episodes=1,
        max_steps_per_ep=max_steps_per_ep,
        device=worker_cfg.device,
        obs_encoder=obs_encoder,
        obs_config={**obs_cfg},
        intrinsic_cfg=intrinsic_cfg,
        frontends_cfg=frontends_cfg,
        gat_cfg=gat_cfg,
        graphs_cfg=graphs_cfg,
        constraints_cfg=constraints_cfg,
        reward_mode=reward_mode,
        reward_source=logging_cfg.get('reward_source', 'raw'),
    )
    return runner


def process_ready_buffers(
    store: PolicyBufferStore,
    policy_map: Dict[str, CategoricalPolicy],
    critic_map: Dict[str, MultiHeadCritic],
    policy_to_critic: Dict[str, str],
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    iter_metrics: Dict[str, Any],
    buffer_meta: Dict[str, Dict[str, Any]],
    gamma: float,
    gae_lambda: float,
    seq_len: int,
    burn_in: int,
    device: torch.device,
    trpo_config: Dict[str, Any],
    xi_time: float,
    xi_batt: float,
    device_str: str,
    scaler: Optional[GradScaler],
    critic_amp_enabled: bool,
    policy_amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, int, List[str]]:
    mask_zero_sum = 0.0
    mask_zero_count = 0
    updated_keys: List[str] = []

    ready = list(store.ready_keys())
    for policy_key in ready:
        buffer = store.get(policy_key)
        meta = buffer_meta.get(policy_key, {})
        if meta:
            buffer.set_obs_metadata({
                'obs_mode': meta.get('obs_mode', 'unknown'),
                'encoder_version': meta.get('obs_encoder_version', 'v1'),
                'obs_dim': meta.get('obs_dim', 0),
            })
        buffer.compute_gae(gamma=gamma, lam=gae_lambda)

        critic_key = policy_to_critic.get(policy_key, next(iter(critic_map.keys())))
        policy_module = policy_map[policy_key]
        critic_module = critic_map[critic_key]
        last_actor_stats: Dict[str, Any] = {}
        last_critic_stats: Dict[str, Any] = {}

        for minibatch in buffer.iter_minibatches(seq_len=seq_len, burn_in=burn_in, device=device):
            mask = minibatch.get('mask')
            if mask is not None:
                mask_zero_sum += float((mask == 0).float().mean().item())
                mask_zero_count += 1

            last_critic_stats = update_critic(
                critic=critic_module,
                optimizer=critic_optimizers[critic_key],
                data_batch=minibatch,
                critic_key=critic_key,
                scaler=scaler if critic_amp_enabled else None,
                amp_enabled=critic_amp_enabled,
                amp_dtype=amp_dtype,
            )
            last_actor_stats = trpo_qp_update_actor(
                policy_module,
                critic_module,
                minibatch,
                xi_time,
                xi_batt,
                trpo_config,
                device_str,
                policy_key=policy_key,
                scaler=scaler if policy_amp_enabled else None,
                amp_enabled=policy_amp_enabled,
                amp_dtype=amp_dtype,
            )

        buffer.clear()
        if last_actor_stats:
            iter_metrics[f"{policy_key}/kl"] = last_actor_stats.get('kl', 0.0)
            iter_metrics[f"{policy_key}/branch"] = last_actor_stats.get('branch', 'N/A')
            iter_metrics[f"{policy_key}/backtracks"] = last_actor_stats.get('backtracks', 0)
        if last_critic_stats:
            iter_metrics[f"{policy_key}/critic_loss"] = last_critic_stats.get('total_loss', 0.0)
        updated_keys.append(policy_key)

    return mask_zero_sum, mask_zero_count, updated_keys


def main():
    parser = argparse.ArgumentParser(description='MOGCRL训练（支持PPO和TRPO-QP）')
    
    # 基本参数
    parser.add_argument('--config', type=str, default='configs/mogcrl.yaml',
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--iters', type=int, default=200, help='训练迭代次数')
    parser.add_argument('--save_dir', type=str, default='results/trpo_qp_test',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu）')
    
    # 约束阈值
    parser.add_argument('--xi_time', type=float, default=-0.2, help='时效约束阈值')
    parser.add_argument('--xi_batt', type=float, default=-0.2, help='电量约束阈值')
    
    # 环境参数
    parser.add_argument('--backend', type=str, default='stub',
                       choices=['stub', 'mock', 'fulfillment', 'rware'],
                       help='环境后端')
    parser.add_argument('--num_agents', type=int, default=2, help='智能体数量')
    parser.add_argument('--episodes', type=int, default=8, help='每轮episode数')
    parser.add_argument('--resume', type=str, default=None, help='从指定 checkpoint 恢复训练')
    parser.add_argument('--graceful_shutdown', action='store_true', help='训练结束时优先发送 DONE 让 workers 优雅退出')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    active_profile = config.get('active_profile')
    if active_profile:
        profile_cfg = (config.get('profiles') or {}).get(active_profile, {})
        if profile_cfg:
            print(f"应用配置 profile: {active_profile}")
            _deep_update(config, profile_cfg)
        else:
            print(f"警告: 未找到 profile '{active_profile}'，使用默认配置")
    
    # 命令行参数覆盖配置
    if args.device is not None:
        device_str = args.device
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化日志
    logger = ExpLogger(
        save_dir=args.save_dir,
        exp_name=f'trpo_qp_seed{args.seed}',
        is_print=True
    )
    
    profile_name = active_profile or 'default'
    constraints_cfg = config.get('constraints', {}) or {}
    if 'xi_time' in constraints_cfg:
        args.xi_time = float(constraints_cfg['xi_time'])
    if 'xi_batt' in constraints_cfg:
        args.xi_batt = float(constraints_cfg['xi_batt'])
    trpo_config = config.get('trpo_qp', {}) or {}
    delta_cfg = float(trpo_config.get('delta', config.get('target_kl', 0.05)))
    accept_ratio_cfg = float(trpo_config.get('accept_ratio', 0.01))
    metadata_line = (
        f"[Setup] seed={args.seed}, profile={profile_name}, "
        f"delta={delta_cfg}, accept_ratio={accept_ratio_cfg}, "
        f"backend={args.backend}, save_dir={args.save_dir}"
    )
    
    print("=" * 70)

    metrics_cfg = config.get('metrics', {}) or {}
    metrics_csv_path = metrics_cfg.get('csv_path')
    metrics_tb_dir = metrics_cfg.get('tb_dir')
    metrics_flush_interval = int(metrics_cfg.get('flush_interval', 10))
    metrics_field_order = metrics_cfg.get('fields')
    tb_writer = None
    if metrics_tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(metrics_tb_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[Metrics] 创建 TensorBoard writer 失败: {exc}")
            tb_writer = None

    eval_cfg = config.get('evaluation', {}) or {}
    eval_enabled = bool(eval_cfg.get('enabled', False))
    eval_every = int(eval_cfg.get('every', 0))
    eval_csv_path = eval_cfg.get('csv_path')
    eval_field_order = eval_cfg.get('fields')
    eval_violation_threshold = float(eval_cfg.get('violation_threshold', 0.0))
    eval_patience = int(eval_cfg.get('patience', 3))
    eval_metric_name = eval_cfg.get('metric', 'eval/return_R_mean')
    eval_higher_better = bool(eval_cfg.get('higher_is_better', True))
    eval_ema_alpha = float(eval_cfg.get('ema_alpha', 0.2))
    eval_ema_delta = float(eval_cfg.get('ema_delta', 1.0))
    best_metric = float('-inf') if eval_higher_better else float('inf')
    metric_ema = None
    violation_streak = 0
    reward_mode = str(config.get('reward_mode', 'engineering_shaping'))
    degrade_streak = 0
    best_ckpt_path: Optional[Path] = None
    best_step: Optional[int] = None

    training_cfg = config.get('training', {}) or {}
    device = torch.device(device_str)
    device_type = device.type
    (
        amp_enabled,
        critic_amp_enabled,
        policy_amp_requested,
        amp_dtype,
        amp_dtype_cfg,
    ) = _resolve_amp_flags(training_cfg, device)
    policy_amp_enabled = False

    ckpt_cfg = training_cfg.get('checkpoint', {}) or {}
    ckpt_dir = Path(ckpt_cfg.get('dir', 'checkpoints'))
    keep_last = int(ckpt_cfg.get('keep_last', 3))
    keep_best = int(ckpt_cfg.get('keep_best', 3))
    best_dir = ckpt_dir / "best"
    best_meta_file = best_dir / "best.json"
    if eval_enabled and best_meta_file.exists():
        try:
            best_info = json.loads(best_meta_file.read_text())
            best_metric = best_info.get('metric', best_metric)
            best_step = best_info.get('step', best_step)
            path_name = best_info.get('path')
            if path_name:
                best_ckpt_candidate = best_dir / path_name
                if best_ckpt_candidate.exists():
                    best_ckpt_path = best_ckpt_candidate.resolve()
        except Exception as exc:
            print(f"[Checkpoint] 读取 {best_meta_file} 失败: {exc}")
    print("开始训练: TRPO-QP 算法")
    print(f"随机种子: {args.seed}")
    print(f"迭代次数: {args.iters}")
    print(f"约束阈值: xi_time={args.xi_time}, xi_batt={args.xi_batt}")
    print(f"设备: {device_str}")
    print(f"基础 seed: {args.seed}")
    device_line = (
        f"[Device] device={device_str}, device_type={device_type}, "
        f"amp.enabled={int(amp_enabled)}, amp.critic={int(critic_amp_enabled)}, "
        f"amp.policy={int(policy_amp_enabled)}, amp.dtype={amp_dtype_cfg}"
    )
    print(device_line)
    print(metadata_line)
    logger.log_fp.write(metadata_line + "\n")
    logger.log_fp.write(device_line + "\n")
    logger.log_fp.flush()
    print("=" * 70)
    
    obs_cfg = config.get('obs', {}) or {}
    use_paper_obs = bool(obs_cfg.get('use_paper_obs', False))
    fov_size = int(obs_cfg.get('fov_size', 3))
    if use_paper_obs:
        obs_encoder = PaperObsEncoder(fov_size=fov_size)
    else:
        obs_encoder = FlatEncoder18D()

    obs_dim = obs_encoder.out_dim
    act_dim = config.get('action_dim', 5)
    
    # 初始化网络
    if policy_amp_requested:
        print("[AMP] 策略路径强制使用 FP32，忽略 policy AMP 设置。")
    scaler = GradScaler(device=device_type, enabled=amp_enabled)
    rnn_cfg = config.get('rnn', {}) or {}
    frontends_cfg = config.get('frontends', {}) or {}
    policy_fe_cfg = frontends_cfg.get('policy', {}) or {}
    critic_fe_cfg = frontends_cfg.get('critic', {}) or {}
    gat_cfg = config.get('gat', {}) or {}
    graphs_cfg = config.get('graphs', {}) or {}
    logging_cfg = config.get('logging', {}) or {}

    use_rnn_policy = bool(policy_fe_cfg.get('use_rnn', rnn_cfg.get('use_rnn_policy', False)))
    use_rnn_critic = bool(critic_fe_cfg.get('use_rnn', rnn_cfg.get('use_rnn', False)))
    policy_use_gat = bool(policy_fe_cfg.get('use_gat', False))
    critic_use_gat = bool(critic_fe_cfg.get('use_gat', False))
    policy_gat_fuse = policy_fe_cfg.get('fuse', 'concat')
    critic_gat_fuse = critic_fe_cfg.get('fuse', 'concat')
    policy_hidden = config.get('policy_hidden', [128, 128])
    critic_hidden = config.get('critic_hidden', [128, 128])

    intrinsic_cfg = config.get('intrinsic', {}) or {}
    use_intrinsic = bool(intrinsic_cfg.get('use_intrinsic', False))
    intrinsic_epsilon = float(intrinsic_cfg.get('epsilon', 0.0)) if use_intrinsic else 0.0
    intrinsic_cfg['use_intrinsic'] = use_intrinsic
    intrinsic_cfg['epsilon'] = intrinsic_epsilon

    policy_args = {
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'hidden': policy_hidden,
        'use_rnn': use_rnn_policy,
        'rnn_cfg': rnn_cfg,
        'use_gat': policy_use_gat,
        'gat_cfg': gat_cfg,
        'gat_fuse': policy_gat_fuse,
    }
    critic_args = {
        'obs_dim': obs_dim,
        'hidden': critic_hidden,
        'agent_embed_dim': 16,
        'use_rnn': use_rnn_critic,
        'rnn_cfg': rnn_cfg,
        'use_gat': critic_use_gat,
        'gat_cfg': gat_cfg,
        'gat_fuse': critic_gat_fuse,
    }
    policy_factory = functools.partial(build_policy_module, device, **policy_args)
    critic_factory = functools.partial(build_critic_module, device, **critic_args)

    policy_map, critic_map, routing = build_registry(
        num_agents=args.num_agents,
        cfg=config,
        make_policy=policy_factory,
        make_critic=critic_factory,
    )
    policy_map = {k: v.to(device) for k, v in policy_map.items()}
    critic_map = {k: v.to(device) for k, v in critic_map.items()}
    primary_policy_key = next(iter(policy_map.keys()))
    primary_critic_key = next(iter(critic_map.keys()))

    irdc_module = None
    if use_intrinsic:
        irdc_module = IRDC(obs_dim=obs_dim, act_dim=act_dim, cfg=intrinsic_cfg).to(device)

    first_linear_policy = None
    for module in policy_map[primary_policy_key].mlp:
        if isinstance(module, torch.nn.Linear):
            first_linear_policy = module
            break
    first_linear_critic = None
    for module in critic_map[primary_critic_key].shared_net:
        if isinstance(module, torch.nn.Linear):
            first_linear_critic = module
            break
    print(f"Encoder output dim: {obs_encoder.out_dim}")
    if first_linear_policy is not None:
        print(f"Policy first layer expects: {first_linear_policy.in_features}")
    if first_linear_critic is not None:
        print(f"Critic first layer expects: {first_linear_critic.in_features}")
    
    critic_optimizers = {
        key: optim.AdamW(module.parameters(), lr=config.get('lr_critic', 6e-4))
        for key, module in critic_map.items()
    }

    updates_since_sync = 0
    start_iter = 0
    broadcast_ver = 0

    resume_path = args.resume
    if resume_path:
        try:
            payload = load_checkpoint(
                resume_path,
                policy_map,
                critic_map,
                critic_optimizers,
                scaler=scaler if amp_enabled else None,
            )
            start_iter = int(payload.get('step', 0)) + 1
            meta_state = payload.get('meta', {}) or {}
            broadcast_ver = meta_state.get('broadcast_ver', broadcast_ver)
            best_metric = meta_state.get('best_metric', best_metric)
            best_step = meta_state.get('best_step', best_step)
            metric_ema = meta_state.get('metric_ema', metric_ema)
            violation_streak = meta_state.get('violation_streak', violation_streak)
            degrade_streak = meta_state.get('degrade_streak', degrade_streak)
            updates_since_sync = meta_state.get('updates_since_sync', updates_since_sync)
            best_path_str = meta_state.get('best_ckpt_path', "")
            if best_path_str:
                candidate = best_dir / best_path_str
                if candidate.exists():
                    best_ckpt_path = candidate.resolve()
                    best_step = meta_state.get('best_step', best_step)
            print(f"[Resume] 从 {resume_path} 恢复，起始迭代 {start_iter}")
        except FileNotFoundError:
            print(f"[Resume] 未找到 {resume_path}，从头开始训练")
            start_iter = 0
            updates_since_sync = 0
            broadcast_ver = 0
    else:
        start_iter = 0

    def checkpoint_meta_state() -> Dict[str, Any]:
        best_name = ""
        resolved_best_dir = best_dir.resolve()
        if isinstance(best_ckpt_path, Path):
            try:
                parent = best_ckpt_path.parent.resolve()
                best_name = best_ckpt_path.name if parent == resolved_best_dir else str(best_ckpt_path)
            except Exception:
                best_name = str(best_ckpt_path)
        elif isinstance(best_ckpt_path, str):
            best_name = best_ckpt_path
        return {
            "broadcast_ver": broadcast_ver,
            "best_metric": best_metric,
            "best_step": best_step,
            "metric_ema": metric_ema,
            "violation_streak": violation_streak,
            "degrade_streak": degrade_streak,
            "updates_since_sync": updates_since_sync,
            "best_ckpt_path": best_name,
        }
    
    # 初始化环境适配器
    backend_adapter = build_backend_adapter(args, config)
    
    # 初始化采样器
    steps_per_agent = config.get('steps_per_agent', 128)
    max_steps_per_ep = config.get('max_steps_per_ep', 256)

    train_cfg = training_cfg
    parallel_cfg = config.get('parallel', {}) or {}
    if device_type == 'cuda' and parallel_cfg.get('worker_device') is None:
        parallel_cfg['worker_device'] = device_str

    policy_to_critic: Dict[str, str] = {}
    for aid, pk in routing.policy_of.items():
        ck = routing.critic_of.get(aid, primary_critic_key)
        policy_to_critic.setdefault(pk, ck)

    def make_buffer(owner_key: str) -> OnPolicyBuffer:
        return OnPolicyBuffer(
            capacity=train_cfg.get('min_timesteps_per_update', 2048),
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_agents=len(routing.policy_of),
            device=device_str,
            obs_mode='paper_obs' if use_paper_obs else 'flat18',
            encoder_version=obs_encoder.__class__.__name__,
            graph_cache_mode=str(graphs_cfg.get('cache_adj_in_buffer', 'dense')).lower(),
            graph_settings={
                'neighbor_mode': gat_cfg.get('neighbor_mode', 'fov'),
                'radius': float(gat_cfg.get('radius', 2.5)),
                'symmetry': graphs_cfg.get('symmetry', 'union'),
                'self_loop': bool(gat_cfg.get('self_loop', True)),
                'topk_neighbors': int(graphs_cfg['topk_neighbors']) if graphs_cfg.get('topk_neighbors') is not None else None,
                'max_degree': graphs_cfg.get('max_degree'),
                'deterministic_topk_jitter': bool(graphs_cfg.get('deterministic_topk_jitter', True)),
            },
            owner_key=owner_key,
            seq_len=train_cfg.get('seq_len', 64),
            burn_in=train_cfg.get('burn_in', 16),
        )

    store = PolicyBufferStore(config, make_buffer)

    num_workers = int(parallel_cfg.get('num_workers', 0))
    queue_timeout = float(parallel_cfg.get('queue_timeout_s', 1.0))
    max_queue_polls = int(parallel_cfg.get('max_queue_polls', 256))
    sync_interval_updates = int(train_cfg.get('sync_interval_updates', 0))
    sync_interval_s = float(train_cfg.get('sync_interval_s', 0.0))
    sync_with_ema = bool(train_cfg.get('sync_with_ema', False))
    ema_tau = float(train_cfg.get('ema_tau', 0.99))
    broadcast_only_backbone = bool(train_cfg.get('broadcast_only_backbone', False))
    updates_total: Dict[str, int] = {pk: 0 for pk in policy_map.keys()}

    runner: Optional[VectorRunner] = None
    if num_workers <= 0:
        runner = VectorRunner(
            backend_adapter=backend_adapter,
            policy=policy_map[primary_policy_key],
            critic=critic_map[policy_to_critic.get(primary_policy_key, primary_critic_key)],
            policy_map=policy_map,
            critic_map=critic_map,
            routing=routing,
            buffer_store=store,
            intrinsic_module=irdc_module,
            steps_per_agent=steps_per_agent,
            num_agents=args.num_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            episodes=args.episodes,
            max_steps_per_ep=max_steps_per_ep,
            device=device_str,
            obs_encoder=obs_encoder,
            obs_config={**obs_cfg},
            intrinsic_cfg=intrinsic_cfg,
            frontends_cfg=frontends_cfg,
            gat_cfg=gat_cfg,
            graphs_cfg=graphs_cfg,
            constraints_cfg=constraints_cfg,
            reward_mode=reward_mode,
            reward_source=logging_cfg.get('reward_source', 'raw'),
        )

    gamma = config.get('gamma', 0.95)
    gae_lambda = config.get('gae_lambda', 0.95)
    rollout_length = parallel_cfg.get('rollout_length', steps_per_agent)
    seq_len = train_cfg.get('seq_len', 64)
    burn_in = train_cfg.get('burn_in', 16)

    last_log_time = time.time()
    samples_accum = 0
    samples_accum_async = 0

    worker_processes: List[Any] = []
    worker_queue = None
    ctrl_conns: List[Any] = []
    ack_versions: Dict[int, int] = {}
    heartbeat_samples: Dict[int, int] = {}
    heartbeat_ts: Dict[int, float] = {}
    last_broadcast_ts = time.time()

    broadcast_policy_map: Optional[Dict[str, CategoricalPolicy]] = None
    broadcast_critic_map: Optional[Dict[str, MultiHeadCritic]] = None

    seed_base = int(parallel_cfg.get('seed_base', config.get('seed', args.seed)))

    if num_workers > 0:
        worker_runner_factory = functools.partial(
            make_worker_runner,
            config=config,
            policy_args=policy_args,
            critic_args=critic_args,
            obs_cfg=obs_cfg,
            frontends_cfg=frontends_cfg,
            gat_cfg=gat_cfg,
            graphs_cfg=graphs_cfg,
            intrinsic_cfg=intrinsic_cfg,
            steps_per_agent=steps_per_agent,
            max_steps_per_ep=max_steps_per_ep,
            logging_cfg=logging_cfg,
            cli_args=args,
        )
        worker_processes, worker_queue, ctrl_conns = start_workers(
            config,
            routing,
            {k: v.state_dict() for k, v in policy_map.items()},
            {k: v.state_dict() for k, v in critic_map.items()},
            runner_factory=worker_runner_factory,
        )
        worker_seeds = [seed_base + idx * 1000 for idx in range(num_workers)]
        print(f"启动异步 workers：num_workers={num_workers}，seed_base={seed_base}，worker_seeds={worker_seeds}")
        ack_versions = {idx: -1 for idx in range(len(ctrl_conns))}
        heartbeat_samples = {idx: 0 for idx in range(len(ctrl_conns))}
        heartbeat_ts = {idx: 0.0 for idx in range(len(ctrl_conns))}

        if sync_with_ema:
            broadcast_policy_map = {
                key: policy_factory()
                for key in policy_map.keys()
            }
            for key, module in broadcast_policy_map.items():
                module.load_state_dict(policy_map[key].state_dict())
            broadcast_critic_map = {
                key: critic_factory()
                for key in critic_map.keys()
            }
            for key, module in broadcast_critic_map.items():
                module.load_state_dict(critic_map[key].state_dict())

    def ema_update_(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p_t, p_s in zip(target.parameters(), source.parameters()):
                p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)

    def gather_broadcast_state(only_backbone: bool = False) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        policy_src = broadcast_policy_map if sync_with_ema and broadcast_policy_map is not None else policy_map
        critic_src = broadcast_critic_map if sync_with_ema and broadcast_critic_map is not None else critic_map
        policy_state = {k: _filter_state_dict(module, only_backbone) for k, module in policy_src.items()}
        critic_state = {k: _filter_state_dict(module, only_backbone) for k, module in critic_src.items()}
        return policy_state, critic_state

    def broadcast_all(use_backbone_only: bool = False) -> None:
        nonlocal broadcast_ver, last_broadcast_ts
        if not ctrl_conns:
            return
        policy_state, critic_state = gather_broadcast_state(use_backbone_only)
        payload = {
            "kind": "BROADCAST",
            "ver": broadcast_ver,
            "policy_state": policy_state,
            "critic_state": critic_state,
        }
        for idx, conn in enumerate(ctrl_conns):
            try:
                conn.send(payload)
            except (BrokenPipeError, EOFError) as exc:
                print(f"[Broadcast] 向 worker {idx} 发送失败: {exc}")
                continue
        broadcast_ver += 1
        last_broadcast_ts = time.time()

    early_stop = False
    stop_reason = ""
    last_iter = start_iter - 1

    try:
        for iter_idx in range(start_iter, args.iters):
            last_iter = iter_idx
            iter_metrics: Dict[str, Any] = {
                'iter': iter_idx,
                'xi_time': args.xi_time,
                'xi_batt': args.xi_batt,
                'intrinsic/use_intrinsic': int(use_intrinsic),
                'intrinsic/epsilon': intrinsic_epsilon,
                'amp/enabled': int(amp_enabled),
                'amp/policy': int(policy_amp_enabled),
                'amp/critic': int(critic_amp_enabled),
                'amp/dtype': str(amp_dtype),
            }
            buffer_meta: Dict[str, Dict[str, Any]] = {}
            pct_mask_zero_total = 0.0
            pct_mask_count = 0
            current_info: Dict[str, Any] = {}
            polled_msgs = 0
            samples_async_iter = 0.0
            updated_keys: List[str] = []
            latency_ms = 0.0

            if num_workers <= 0 and runner is not None:
                info_agg, _ = runner.rollout_multi_policy_sync(rollout_length, store_batches=True)
                samples_accum += int(info_agg.get('samples_collected', 0))
                buffer_meta = {pk: info_agg for pk in policy_map.keys()}
                current_info = info_agg
                t0 = time.perf_counter()
                mask_sum, mask_cnt, updated_keys = process_ready_buffers(
                    store,
                    policy_map,
                    critic_map,
                    policy_to_critic,
                    critic_optimizers,
                    iter_metrics,
                    buffer_meta,
                    gamma,
                    gae_lambda,
                    seq_len,
                    burn_in,
                    device,
                    trpo_config,
                    args.xi_time,
                    args.xi_batt,
                    device_str,
                    scaler,
                    critic_amp_enabled,
                    policy_amp_enabled,
                    amp_dtype,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                for pk in updated_keys:
                    updates_total[pk] += 1
                updates_since_sync += len(updated_keys)
                pct_mask_zero_total += mask_sum
                pct_mask_count += mask_cnt
            else:
                assert worker_queue is not None
                deadline = time.time() + queue_timeout
                polls = 0
                while time.time() < deadline:
                    try:
                        msg = worker_queue.get_nowait()
                    except queue.Empty:
                        break
                    polls += 1
                    if polls >= max_queue_polls:
                        break
                    polled_msgs += 1
                    if not isinstance(msg, dict):
                        continue
                    kind = msg.get('kind')
                    if kind == 'ROLL':
                        pk = msg['policy_key']
                        batch_cpu = msg.get('batch')
                        if not (batch_cpu and isinstance(batch_cpu, dict)):
                            continue
                        obs_tensor = batch_cpu.get('obs')
                        if not (isinstance(obs_tensor, torch.Tensor) and obs_tensor.numel() > 0):
                            continue
                        if obs_tensor.shape[0] == 0 or obs_tensor.shape[1] == 0:
                            continue
                        store.add_msg(pk, batch_cpu)
                        samples_inc = int(obs_tensor.shape[0] * obs_tensor.shape[1])
                        samples_accum += samples_inc
                        samples_accum_async += samples_inc
                        samples_async_iter += samples_inc
                        info_payload = msg.get('info')
                        if info_payload:
                            buffer_meta[pk] = info_payload
                            current_info = info_payload
                    elif kind == 'ACK':
                        worker_id = msg.get('worker_id')
                        if isinstance(worker_id, int):
                            ack_versions[worker_id] = int(msg.get('ver', ack_versions.get(worker_id, -1)))
                    elif kind == 'HEARTBEAT':
                        worker_id = msg.get('worker_id')
                        if isinstance(worker_id, int):
                            heartbeat_samples[worker_id] = int(msg.get('samples', 0))
                            heartbeat_ts[worker_id] = float(msg.get('ts', time.time()))
                    elif kind == 'LOAD_FAIL':
                        print(f"[Worker {msg.get('worker_id')}] load_state 失败 ver={msg.get('ver')} exc={msg.get('exc')}")
                    elif kind == 'WORKER_CRASH':
                        print(f"[Worker {msg.get('worker_id')}] 崩溃: {msg.get('exc')}")
                        iter_metrics.setdefault('warnings/worker_crash', 0)
                        iter_metrics['warnings/worker_crash'] += 1
                    elif kind == 'DONE_ACK':
                        # 仅在优雅退出时使用，这里忽略
                        continue
                    else:
                        continue

                t0 = time.perf_counter()
                mask_sum, mask_cnt, updated_keys = process_ready_buffers(
                    store,
                    policy_map,
                    critic_map,
                    policy_to_critic,
                    critic_optimizers,
                    iter_metrics,
                    buffer_meta,
                    gamma,
                    gae_lambda,
                    seq_len,
                    burn_in,
                    device,
                    trpo_config,
                    args.xi_time,
                    args.xi_batt,
                    device_str,
                    scaler,
                    critic_amp_enabled,
                    policy_amp_enabled,
                    amp_dtype,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                for pk in updated_keys:
                    updates_total[pk] += 1
                pct_mask_zero_total += mask_sum
                pct_mask_count += mask_cnt
                updates_since_sync += len(updated_keys)
            iter_metrics['parallel/polled_msgs'] = polled_msgs
            iter_metrics['samples_async/s'] = (samples_async_iter / max(queue_timeout, 1e-6)) if num_workers > 0 else 0.0
            iter_metrics['update_latency_ms'] = latency_ms

            if sync_with_ema and broadcast_policy_map is not None and broadcast_critic_map is not None and updated_keys:
                for pk in updated_keys:
                    if pk in broadcast_policy_map:
                        ema_update_(broadcast_policy_map[pk], policy_map[pk], ema_tau)
                    critic_key = policy_to_critic.get(pk, primary_critic_key)
                    if critic_key in broadcast_critic_map:
                        ema_update_(broadcast_critic_map[critic_key], critic_map[critic_key], ema_tau)

            if ctrl_conns:
                if sync_interval_updates > 0 and updates_since_sync >= sync_interval_updates:
                    broadcast_all(broadcast_only_backbone)
                    updates_since_sync = 0
                if sync_interval_s > 0 and (time.time() - last_broadcast_ts) >= sync_interval_s:
                    broadcast_all(broadcast_only_backbone)
                    updates_since_sync = 0
                record_worker_health(
                    iter_metrics,
                    ack_versions,
                    broadcast_ver,
                    heartbeat_ts,
                    time.time(),
                    queue_timeout,
                )

            iter_metrics.setdefault('parallel/polled_msgs', polled_msgs)

            env_metrics = aggregate_env_metrics([current_info]) if current_info else {}
            iter_metrics.update({
                'obs_mode': current_info.get('obs_mode', 'unknown'),
                'obs_dim': current_info.get('obs_dim', obs_dim),
                'uses_paper_obs': current_info.get('uses_paper_obs', int(use_paper_obs)),
                'obs_fallback': current_info.get('obs_fallback', 0),
                'obs_encoder_version': current_info.get('obs_encoder_version', obs_encoder.__class__.__name__),
                'gat/avg_degree': current_info.get('gat/avg_degree', 0.0),
                'gat/pct_self_only': current_info.get('gat/pct_self_only', 0.0),
                'gat/neighbor_mode': current_info.get('gat/neighbor_mode', gat_cfg.get('neighbor_mode', 'fov')),
                **env_metrics,
            })

            episode_stats = {
                'avg_reward_per_agent': env_metrics.get('avg_reward_per_agent', 0.0),
                'avg_cost_time_per_agent': env_metrics.get('avg_cost_time_per_agent', 0.0),
                'avg_cost_batt_per_agent': env_metrics.get('avg_cost_batt_per_agent', 0.0),
                'cost_sum_per_agent': env_metrics.get('cost_sum_per_agent', 0.0),
                'total_reward_sum': env_metrics.get('total_reward_sum', 0.0),
                'total_cost_time_sum': env_metrics.get('total_cost_time_sum', 0.0),
                'total_cost_batt_sum': env_metrics.get('total_cost_batt_sum', 0.0),
                'total_constraint_cost_sum': env_metrics.get('total_constraint_cost_sum', 0.0),
                'episode_steps_total': env_metrics.get('episode_steps_total', 0.0),
                'episode_length_avg': env_metrics.get('episode_length_avg', 0.0),
                'episode_length_last': env_metrics.get('episode_length_last', 0.0),
                'total_agent_steps': env_metrics.get('total_agent_steps', 0.0),
                'viol_time_rate': env_metrics.get('viol_time_rate', 0.0),
                'viol_batt_rate': env_metrics.get('viol_batt_rate', 0.0),
                'debug_raw_reward_sum': env_metrics.get('debug_raw_reward_sum', 0.0),
                'debug_raw_available': float(env_metrics.get('debug_raw_available', False)),
                'debug_raw_present_steps': env_metrics.get('debug_raw_present_steps', 0.0),
                'debug_raw_finite_ratio': env_metrics.get('debug_raw_finite_ratio', 0.0),
                'debug_rew_shape_T': env_metrics.get('debug_rew_shape_T', 0.0),
                'debug_rew_shape_B': env_metrics.get('debug_rew_shape_B', 0.0),
                'debug_num_agents': env_metrics.get('debug_num_agents', 0.0),
                'debug_rollout_length': env_metrics.get('debug_rollout_length', 0.0),
                'debug_samples_collected': env_metrics.get('debug_samples_collected', 0.0),
                'reward_source_used': env_metrics.get('reward_source_used', 'normalized'),
            }
            logger.log_episode_stats(iter_idx, episode_stats)

            if use_intrinsic:
                steps_int = max(1, current_info.get('intrinsic_steps', 1))
                iter_metrics['intrinsic/r_int_mean'] = current_info.get('r_int_sum', 0.0) / steps_int
                iter_metrics['intrinsic/r_mix_mean'] = current_info.get('r_mix_sum', 0.0) / steps_int
            else:
                iter_metrics['intrinsic/r_int_mean'] = 0.0
                iter_metrics['intrinsic/r_mix_mean'] = 0.0

            if pct_mask_count > 0:
                iter_metrics['pct_mask_zero'] = pct_mask_zero_total / pct_mask_count
            else:
                iter_metrics['pct_mask_zero'] = 0.0

            if eval_enabled and eval_every > 0 and (iter_idx + 1) % eval_every == 0:
                eval_metrics = run_evaluation(
                    policy_map,
                    critic_map,
                    routing,
                    eval_cfg,
                    args,
                    config,
                    obs_encoder,
                    irdc_module,
                    frontends_cfg,
                    gat_cfg,
                    graphs_cfg,
                    device_str,
                    steps_per_agent,
                    max_steps_per_ep,
                    primary_policy_key,
                    primary_critic_key,
                    logging_cfg.get('reward_source', 'raw'),
                )
                iter_metrics.update(eval_metrics)
                if eval_csv_path:
                    if eval_field_order is None:
                        eval_field_order = sorted(eval_metrics.keys())
                    export_csv(eval_metrics, eval_csv_path, eval_field_order)
                metric_value = eval_metrics.get(eval_metric_name)
                violation_value = eval_metrics.get('eval/viol_batt_mean', 0.0)
                if metric_value is not None:
                    if metric_ema is None:
                        metric_ema = metric_value
                    else:
                        metric_ema = eval_ema_alpha * metric_value + (1.0 - eval_ema_alpha) * metric_ema
                    improved = (eval_higher_better and metric_value > best_metric) or (not eval_higher_better and metric_value < best_metric)
                    if improved:
                        best_metric = metric_value
                        best_step = iter_idx
                        violation_streak = 0
                        degrade_streak = 0
                        best_ckpt_path = (best_dir / f"best_step{iter_idx:08d}.pt").resolve()
                        save_checkpoint(
                            best_dir,
                            'best',
                            iter_idx,
                            policy_map,
                            critic_map,
                            critic_optimizers,
                            scaler=scaler if amp_enabled else None,
                            meta=checkpoint_meta_state(),
                        )
                        cleanup_checkpoints(best_dir, 'best', keep_best)
                        save_best_meta(best_dir, {"step": iter_idx, "metric": best_metric, "path": best_ckpt_path.name})
                        print(f"[Eval] step {iter_idx}: {eval_metric_name} 提升至 {metric_value:.4f}")
                    else:
                        degrade = False
                        if metric_ema is not None:
                            if eval_higher_better:
                                degrade = metric_value < (metric_ema - eval_ema_delta)
                            else:
                                degrade = metric_value > (metric_ema + eval_ema_delta)
                        degrade_streak = degrade_streak + 1 if degrade else 0
                        violation_streak = violation_streak + 1 if violation_value > eval_violation_threshold else 0
                        if max(violation_streak, degrade_streak) >= eval_patience:
                            print(f"[Eval] 触发早停：viol={violation_streak}, degrade={degrade_streak}")
                            if best_ckpt_path and best_ckpt_path.exists():
                                load_checkpoint(
                                    best_ckpt_path,
                                    policy_map,
                                    critic_map,
                                    critic_optimizers,
                                    scaler=scaler if amp_enabled else None,
                                )
                            early_stop = True
                            stop_reason = "evaluation_patience"
                            break

            iter_metrics['training/broadcast_only_backbone'] = int(broadcast_only_backbone)
            iter_metrics['parallel/num_workers'] = num_workers
            iter_metrics['broadcast/ver'] = broadcast_ver
            if ctrl_conns:
                for wid in range(len(ctrl_conns)):
                    iter_metrics[f'worker{wid}/ack_ver'] = ack_versions.get(wid, -1)
                    iter_metrics[f'worker{wid}/hb_samples'] = heartbeat_samples.get(wid, 0)
                    iter_metrics[f'worker{wid}/hb_ts'] = heartbeat_ts.get(wid, 0.0)
            iter_metrics['training/ema_tau'] = ema_tau if sync_with_ema else 0.0
            for key, value in iter_metrics.items():
                if isinstance(value, float) and np.isnan(value):
                    iter_metrics[key] = 0.0

            logger.log_dict(iter_metrics)

            should_flush_metrics = metrics_flush_interval <= 1 or (iter_idx % metrics_flush_interval == 0)
            if metrics_csv_path and should_flush_metrics:
                if metrics_field_order is None:
                    metrics_field_order = sorted(iter_metrics.keys())
                export_csv(iter_metrics, metrics_csv_path, metrics_field_order)
            if tb_writer and should_flush_metrics:
                export_tb(iter_metrics, tb_writer, iter_idx)

            meta_state = checkpoint_meta_state()
            save_checkpoint(
                ckpt_dir,
                'ckpt',
                iter_idx,
                policy_map,
                critic_map,
                critic_optimizers,
                scaler=scaler if amp_enabled else None,
                meta=meta_state,
            )
            cleanup_checkpoints(ckpt_dir, 'ckpt', keep_last)

            now = time.time()
            if now - last_log_time > 2.0:
                policy_state_log, critic_state_log = gather_broadcast_state(broadcast_only_backbone)
                total_bytes = 0
                for state in list(policy_state_log.values()) + list(critic_state_log.values()):
                    for tensor in state.values():
                        if isinstance(tensor, torch.Tensor):
                            total_bytes += tensor.numel() * tensor.element_size()
                log_payload = {
                    'samples/s': samples_accum / max(1e-6, now - last_log_time),
                    'samples_async/s': samples_accum_async / max(1e-6, now - last_log_time),
                    'parallel/num_workers': num_workers,
                    'broadcast/ver': broadcast_ver,
                    'broadcast/bytes': total_bytes,
                }
                logger.log_dict(log_payload)
                samples_accum = 0
                samples_accum_async = 0
                last_log_time = now

            if early_stop:
                break

    finally:
        if ctrl_conns:
            if args.graceful_shutdown and worker_queue is not None:
                pending_done = set(range(len(ctrl_conns)))
                for conn in ctrl_conns:
                    try:
                        conn.send({"kind": "DONE"})
                    except (BrokenPipeError, EOFError):
                        continue
                shutdown_deadline = time.time() + max(queue_timeout, 1.0)
                while pending_done and time.time() < shutdown_deadline:
                    try:
                        msg = worker_queue.get(timeout=max(0.0, shutdown_deadline - time.time()))
                    except queue.Empty:
                        break
                    if isinstance(msg, dict) and msg.get("kind") == "DONE_ACK":
                        wid = msg.get("worker_id")
                        if isinstance(wid, int) and wid in pending_done:
                            pending_done.discard(wid)
                if pending_done:
                    print(f"[Shutdown] 未收到 DONE_ACK 的 workers: {sorted(pending_done)}")
            for conn in ctrl_conns:
                try:
                    conn.close()
                except Exception:
                    pass
        if worker_processes:
            for proc in worker_processes:
                proc.terminate()
                proc.join()

    if early_stop:
        print(f"[Train] 提前结束训练，原因: {stop_reason}")

    final_meta = checkpoint_meta_state()
    save_checkpoint(
        ckpt_dir,
        'final',
        last_iter,
        policy_map,
        critic_map,
        critic_optimizers,
        scaler=scaler if amp_enabled else None,
        meta=final_meta,
    )
    cleanup_checkpoints(ckpt_dir, 'final', 1)

    model_path = Path(args.save_dir) / f'trpo_qp_seed{args.seed}.pt'
    torch.save({
        'policy_state_dict': {k: v.state_dict() for k, v in policy_map.items()},
        'critic_state_dict': {k: v.state_dict() for k, v in critic_map.items()},
        'optimizer_state_dict': {k: opt.state_dict() for k, opt in critic_optimizers.items()},
        'config': config,
        'meta': final_meta,
    }, model_path)

    if tb_writer:
        tb_writer.flush()
        tb_writer.close()

    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"模型已保存到 {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
