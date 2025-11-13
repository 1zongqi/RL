"""
Asynchronous rollout workers for multi-policy sampling.

Workers run in separate processes, collect rollouts using a VectorRunner-like
factory, bucket trajectories by policy_key, and send CPU tensors back to the
collector through a multiprocessing queue.
"""

from __future__ import annotations

import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import multiprocessing as mp

from ..agents import AgentRouting


def _clone_optional(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value

PolicyKey = str
Batch = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: int
    num_envs: int
    rollout_length: int
    device: str
    pin_memory: bool = True
    queue_maxsize: int = 8
    seed: Optional[int] = None
    max_queue_wait_s: float = 10.0


@dataclass
class RolloutPackage:
    policy_key: PolicyKey
    batch: Batch
    meta: Dict[str, torch.Tensor]
    worker_id: int


def _ensure_spawn_start() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set – ignore
        pass


def _pin_if_needed(tensor: torch.Tensor, pin: bool) -> torch.Tensor:
    if pin and tensor.device.type == "cpu" and not tensor.is_pinned():
        return tensor.pin_memory()
    return tensor


def _move_batch_to_cpu(batch: Batch, pin_memory: bool) -> Batch:
    cpu_batch: Batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            cpu_batch[key] = _pin_if_needed(tensor, pin_memory)
        else:
            cpu_batch[key] = value
    return cpu_batch


def _bucketize_rollout(
    rollout: Dict[str, torch.Tensor],
    routing: AgentRouting,
) -> Dict[PolicyKey, Batch]:
    policy_to_agents: Dict[str, list] = {}
    for agent_id, key in routing.policy_of.items():
        policy_to_agents.setdefault(key, []).append(agent_id)

    time_steps = int(rollout.get("time_steps", 0))
    num_agents = int(rollout.get("num_agents", 0))
    if time_steps <= 0 or num_agents <= 0:
        raise ValueError("rollout missing time_steps/num_agents metadata")

    obs = rollout['obs'].view(time_steps, num_agents, -1)
    actions = rollout['act'].view(time_steps, num_agents)
    logp = rollout['logp'].view(time_steps, num_agents)
    logits = rollout['logits'].view(time_steps, num_agents, -1)
    mask = rollout['mask'].view(time_steps, num_agents)
    done = rollout['done'].view(time_steps, num_agents)

    values_dict = rollout['values_dict']
    v_R = values_dict['R'].view(time_steps, num_agents, -1)
    v_time = values_dict['time'].view(time_steps, num_agents, -1)
    v_batt = values_dict['batt'].view(time_steps, num_agents, -1)

    rew_dict = rollout['rew_dict']
    r_ext = rew_dict['R'].view(time_steps, num_agents)
    r_time = rew_dict['time'].view(time_steps, num_agents)
    r_batt = rew_dict['batt'].view(time_steps, num_agents)
    r_int = rew_dict.get('int', torch.zeros_like(r_ext)).view(time_steps, num_agents)
    r_mix = rew_dict.get('mix', r_ext).view(time_steps, num_agents)

    adj = rollout.get('adj')
    if isinstance(adj, torch.Tensor):
        adj = adj.view(time_steps, num_agents, num_agents)

    batches: Dict[PolicyKey, Batch] = {}
    for policy_key, agent_ids in policy_to_agents.items():
        agent_ids_sorted = sorted(agent_ids)
        idx = torch.tensor(agent_ids_sorted, dtype=torch.long)
        batches[policy_key] = {
            "obs": obs[:, idx, :],
            "actions": actions[:, idx],
            "logp": logp[:, idx],
            "logits": logits[:, idx, :],
            "mask": mask[:, idx],
            "done": done[:, idx],
            "v_R": v_R[:, idx, :],
            "v_time": v_time[:, idx, :],
            "v_batt": v_batt[:, idx, :],
            "reward_ext": r_ext[:, idx],
            "reward_time": r_time[:, idx],
            "reward_batt": r_batt[:, idx],
            "reward_int": r_int[:, idx],
            "reward_mix": r_mix[:, idx],
            "agent_ids": idx,
            "h0_policy": _clone_optional(rollout.get("h0_policy")),
            "h0_critic": _clone_optional(rollout.get("h0_critic")),
        }
        if isinstance(adj, torch.Tensor):
            batches[policy_key]["adj"] = adj[:, idx][:, :, idx].clone()
    return batches


def _worker_loop(
    worker_cfg: WorkerConfig,
    routing: AgentRouting,
    rollout_fn: Callable[[int], Dict[str, torch.Tensor]],
    out_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    torch.manual_seed(worker_cfg.seed or 0)
    if hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(worker_cfg.seed or 0)

    while not stop_event.is_set():
        batch = rollout_fn(worker_cfg.worker_id)
        if batch is None:
            continue
        buckets = _bucketize_rollout(batch, routing)
        timestamp = time.time()
        for policy_key, payload in buckets.items():
            package = RolloutPackage(
                policy_key=policy_key,
                batch=_move_batch_to_cpu(payload, worker_cfg.pin_memory),
                meta={"timestamp": torch.tensor(timestamp)},
                worker_id=worker_cfg.worker_id,
            )
            _put_with_backpressure(out_queue, package, worker_cfg.max_queue_wait_s)


def _put_with_backpressure(queue_obj: mp.Queue, item: RolloutPackage, timeout_s: float) -> None:
    start = time.time()
    while True:
        try:
            queue_obj.put(item, timeout=timeout_s)
            return
        except queue.Full:
            if time.time() - start > timeout_s:
                raise TimeoutError("queue put timed out; collector not draining fast enough")


class AsyncWorkerManager:
    """
    Manage a pool of asynchronous rollout workers.

    Parameters
    ----------
    worker_cfgs:
        Configuration per worker.
    routing:
        Agent routing defining policy/critic assignments.
    runner_factory:
        Callable taking (worker_id, worker_cfg) and returning a rollout dict
        when invoked repeatedly. The callable must be picklable.
    """

    def __init__(
        self,
        worker_cfgs: Iterable[WorkerConfig],
        routing: AgentRouting,
        runner_factory: Callable[[int, WorkerConfig], Callable[[int], Dict[str, torch.Tensor]]],
    ) -> None:
        _ensure_spawn_start()
        self._queue: mp.Queue = mp.Queue(maxsize=max(cfg.queue_maxsize for cfg in worker_cfgs))
        self._stop_event = mp.Event()
        self._processes: Dict[int, mp.Process] = {}
        self._routing = routing

        for cfg in worker_cfgs:
            rollout_fn = runner_factory(cfg.worker_id, cfg)
            proc = mp.Process(
                target=_worker_loop,
                args=(cfg, routing, rollout_fn, self._queue, self._stop_event),
                daemon=True,
            )
            proc.start()
            self._processes[cfg.worker_id] = proc

    def get(self, timeout: Optional[float] = None) -> RolloutPackage:
        return self._queue.get(timeout=timeout)

    def close(self) -> None:
        self._stop_event.set()
        for proc in self._processes.values():
            proc.join(timeout=5)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _simple_worker_entry(
    worker_cfg: WorkerConfig,
    routing: AgentRouting,
    policy_state: Dict[str, Dict[str, Any]],
    critic_state: Dict[str, Dict[str, Any]],
    runner_factory: Callable[[WorkerConfig, AgentRouting, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]], Any],
    out_queue: mp.Queue,
    ctrl_conn,
) -> None:
    worker_seed = worker_cfg.seed or 0
    torch.manual_seed(worker_seed)
    if hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except Exception:
        pass
    try:
        import random
        random.seed(worker_seed)
    except Exception:
        pass
    runner = runner_factory(worker_cfg, routing, policy_state, critic_state)
    rollout_length = worker_cfg.rollout_length
    local_version = -1

    while True:
        try:
            if ctrl_conn is not None and ctrl_conn.poll():
                ctrl_msg = ctrl_conn.recv()
                if not isinstance(ctrl_msg, dict):
                    continue
                kind = ctrl_msg.get("kind")
                if kind == "BROADCAST":
                    try:
                        runner.load_state(
                            policy_state=ctrl_msg.get("policy_state"),
                            critic_state=ctrl_msg.get("critic_state"),
                        )
                        local_version = int(ctrl_msg.get("ver", local_version + 1))
                        ack_msg = {
                            "kind": "ACK",
                            "worker_id": worker_cfg.worker_id,
                            "ver": local_version,
                            "ts": time.time(),
                        }
                        _put_with_backpressure(out_queue, ack_msg, worker_cfg.max_queue_wait_s)
                    except Exception as exc:  # noqa: BLE001
                        warn_msg = {
                            "kind": "LOAD_FAIL",
                            "worker_id": worker_cfg.worker_id,
                            "ver": ctrl_msg.get("ver"),
                            "exc": str(exc),
                            "tb": traceback.format_exc(),
                        }
                        _put_with_backpressure(out_queue, warn_msg, worker_cfg.max_queue_wait_s)
                elif kind == "DONE":
                    done_msg = {
                        "kind": "DONE_ACK",
                        "worker_id": worker_cfg.worker_id,
                        "ts": time.time(),
                    }
                    _put_with_backpressure(out_queue, done_msg, worker_cfg.max_queue_wait_s)
                    break

            info_dict, batches = runner.rollout_multi_policy_sync(rollout_length, store_batches=False)
            info_sent = False
            samples_total = 0
            for policy_key, batch_cpu in batches.items():
                message = {
                    "kind": "ROLL",
                    "policy_key": policy_key,
                    "batch": batch_cpu,
                }
                if not info_sent:
                    message["info"] = info_dict
                    info_sent = True
                obs_tensor = batch_cpu.get("obs") if isinstance(batch_cpu, dict) else None
                if isinstance(obs_tensor, torch.Tensor):
                    samples_total += int(obs_tensor.shape[0] * obs_tensor.shape[1])
                _put_with_backpressure(out_queue, message, worker_cfg.max_queue_wait_s)

            heartbeat = {
                "kind": "HEARTBEAT",
                "worker_id": worker_cfg.worker_id,
                "samples": samples_total,
                "ts": time.time(),
                "ver": local_version,
            }
            _put_with_backpressure(out_queue, heartbeat, worker_cfg.max_queue_wait_s)
        except Exception as exc:  # noqa: BLE001
            crash_msg = {
                "kind": "WORKER_CRASH",
                "worker_id": worker_cfg.worker_id,
                "exc": str(exc),
                "tb": traceback.format_exc(),
            }
            try:
                _put_with_backpressure(out_queue, crash_msg, worker_cfg.max_queue_wait_s)
            finally:
                break


def start_workers(
    cfg: Dict[str, Any],
    routing: AgentRouting,
    policy_state: Dict[str, Dict[str, Any]],
    critic_state: Dict[str, Dict[str, Any]],
    runner_factory: Callable[[WorkerConfig, AgentRouting, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]], Any],
) -> Tuple[List[mp.Process], mp.Queue, List[Any]]:
    parallel_cfg = cfg.get('parallel', {}) or {}
    num_workers = int(parallel_cfg.get('num_workers', 1))
    rollout_length = int(parallel_cfg.get('rollout_length', parallel_cfg.get('rollout_len', 64)))
    queue_capacity = max(2, int(parallel_cfg.get('queue_maxsize', max(1, num_workers) * 2)))
    queue_timeout = float(parallel_cfg.get('queue_timeout_s', 1.0))
    pin_memory = bool(parallel_cfg.get('pin_memory', True))
    seed_base = int(parallel_cfg.get('seed_base', cfg.get('seed', 0) or 0))
    worker_device = str(parallel_cfg.get('worker_device', 'cpu'))

    worker_cfgs = [
        WorkerConfig(
            worker_id=wid,
            num_envs=int(parallel_cfg.get('num_envs', 1)),
            rollout_length=rollout_length,
            device=worker_device,
            pin_memory=pin_memory,
            queue_maxsize=queue_capacity,
            seed=seed_base + wid * 1000,
            max_queue_wait_s=queue_timeout,
        )
        for wid in range(num_workers)
    ]

    _ensure_spawn_start()
    out_queue: mp.Queue = mp.Queue(maxsize=queue_capacity)
    processes: List[mp.Process] = []
    parent_conns: List[Any] = []
    child_conns: List[Any] = []

    for _ in range(num_workers):
        c_conn, p_conn = mp.Pipe(duplex=False)
        parent_conns.append(p_conn)
        child_conns.append(c_conn)

    for worker_cfg, ctrl_conn in zip(worker_cfgs, child_conns):
        proc = mp.Process(
            target=_simple_worker_entry,
            args=(worker_cfg, routing, policy_state, critic_state, runner_factory, out_queue, ctrl_conn),
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    return processes, out_queue, parent_conns

