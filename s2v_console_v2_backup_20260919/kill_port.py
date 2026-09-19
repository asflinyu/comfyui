#!/usr/bin/env python3
"""关闭占用指定端口的进程。"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path


def find_pids_by_port(port: int) -> list[int]:
    """通过 /proc/net/tcp 查找占用 port 的进程 PID。"""
    port_hex = f"{port:04X}"
    inode: str | None = None
    try:
        with open("/proc/net/tcp") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local_addr = parts[1]
                # local_addr 格式: 0A00001F:1F90 -> host:port_hex
                if not local_addr.endswith(f":{port_hex}"):
                    continue
                inode = parts[9]
                if inode == "0":
                    inode = None
                break
    except FileNotFoundError:
        print("错误: /proc/net/tcp 不存在，本工具仅支持 Linux")
        return []

    if not inode:
        return []

    pids: set[int] = set()
    for fd_link in Path("/proc").glob("[0-9]*/fd/[0-9]*"):
        try:
            target = os.readlink(fd_link)
        except OSError:
            continue
        if target == f"socket:[{inode}]":
            pid = int(fd_link.parts[2])
            if pid != os.getpid() and pid != os.getppid():
                pids.add(pid)
    return sorted(pids)


def kill_pids(pids: list[int], port: int, force: bool = False) -> None:
    """结束指定 PID，先优雅终止，超时后强制杀死。"""
    sig = signal.SIGKILL if force else signal.SIGTERM
    for pid in pids:
        try:
            os.kill(pid, sig)
            print(f"已发送 {'SIGKILL' if force else 'SIGTERM'} 到 PID {pid}")
        except ProcessLookupError:
            pass

    if not force:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            still_alive = [pid for pid in pids if _is_alive(pid)]
            if not still_alive:
                return
            time.sleep(0.2)
        print(f"端口 {port} 的进程未退出，执行强制结束...")
        kill_pids(pids, port, force=True)


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="关闭占用指定 TCP 端口的进程")
    parser.add_argument("port", type=int, nargs="?", default=6010, help="目标端口，默认 6010")
    parser.add_argument("--force", "-f", action="store_true", help="直接 SIGKILL，不等优雅退出")
    args = parser.parse_args()

    pids = find_pids_by_port(args.port)
    if not pids:
        print(f"端口 {args.port} 未被占用")
        sys.exit(0)

    print(f"端口 {args.port} 被以下进程占用: {pids}")
    kill_pids(pids, args.port, force=args.force)

    remaining = [pid for pid in pids if _is_alive(pid)]
    if remaining:
        print(f"失败: 仍有进程存活 {remaining}")
        sys.exit(1)
    print(f"端口 {args.port} 已释放")


if __name__ == "__main__":
    main()
