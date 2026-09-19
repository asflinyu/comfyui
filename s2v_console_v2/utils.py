# coding=utf-8
"""通用工具函数。"""
from __future__ import annotations


def debug_break() -> None:
    """只在 VS Code debugpy 调试器连接时才中断。

    用途：替代裸 `breakpoint()`，避免 `./start.sh` 等生产启动方式被 pdb 卡住。
    """
    import sys

    # VS Code F5 启动时会由 debugpy launcher 注入 debugpy 模块；
    # 生产环境未加载 debugpy，直接跳过，避免 import 开销和警告。
    if "debugpy" not in sys.modules:
        return
    try:
        import debugpy

        if debugpy.is_client_connected():
            debugpy.breakpoint()
    except Exception:
        pass
