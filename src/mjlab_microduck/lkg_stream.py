"""Optional LKG live-stream: publish env-0 qpos over UDP on every env step.

Opt-in: set ``MICRODUCK_LKG_STREAM=<port>`` (panel default 44123). The
mujoco-lkg viewer (`--stream <port>`) receives the packets, poses its own
classic-MuJoCo copy of the robot and renders the true light-field quilt —
live training progress on the Looking Glass.

Observational only: when the env var is unset nothing is constructed, and
every failure inside the publish path is swallowed — the display channel
must never perturb training (sim2real invariants untouched: no state is
written, no RNG consumed).
"""

from __future__ import annotations

import socket
import struct

_MAGIC = 0x4C4B4751  # 'LKGQ'
_PACKET_HDR = "<IIf"  # magic, nq, sim_t


def _base_env(env):
    """Walk .env / .unwrapped wrapper chains down to ManagerBasedRlEnv (.sim)."""
    seen = 0
    while not hasattr(env, "sim"):
        nxt = getattr(env, "env", None) or getattr(env, "unwrapped", None)
        if nxt is None or nxt is env or seen > 16:
            raise AttributeError("no ManagerBasedRlEnv with .sim in wrapper chain")
        env = nxt
        seen += 1
    return env


def _to_np(arr):
    """warp array → numpy（CPU 上零拷贝）；容忍已是 numpy 的测试替身。"""
    return arr.numpy() if hasattr(arr, "numpy") else __import__("numpy").asarray(arr)


class LKGStreamEnvProxy:
    """Duck-typed proxy around RslRlVecEnvWrapper.

    Forwards every attribute to the wrapped env; after each ``step()`` it
    publishes env-0's qpos (and sim time) as one small UDP datagram.
    """

    def __init__(self, env, port: int, env_index: int = 0):
        object.__setattr__(self, "_lkg_env", env)
        object.__setattr__(self, "_lkg_dest", ("127.0.0.1", int(port)))
        object.__setattr__(self, "_lkg_idx", int(env_index))
        object.__setattr__(self, "_lkg_sock", socket.socket(socket.AF_INET, socket.SOCK_DGRAM))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_lkg_env"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_lkg_env"), name, value)

    def step(self, actions):
        out = self._lkg_env.step(actions)
        try:
            self._publish()
        except Exception:
            pass  # display-only channel; never let it break a run
        return out

    def _publish(self):
        data = _base_env(self._lkg_env).sim.data
        qpos = _to_np(data.qpos)  # (nworld, nq); CPU warp is numpy-backed
        row = qpos[self._lkg_idx]
        time_arr = getattr(data, "time", None)
        sim_t = float(_to_np(time_arr)[self._lkg_idx]) if time_arr is not None else 0.0
        pkt = struct.pack(_PACKET_HDR, _MAGIC, len(row), sim_t) + row.astype("<f4").tobytes()
        self._lkg_sock.sendto(pkt, self._lkg_dest)
