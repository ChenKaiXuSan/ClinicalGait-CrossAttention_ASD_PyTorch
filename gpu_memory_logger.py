import logging
import pynvml
from pytorch_lightning.callbacks import Callback


class GpuMemoryLogger(Callback):
    """
    在每个 epoch 结束后统计 GPU 显存使用量与占比，
    同时写入 Lightning logger 与 Python logger。
    """

    def __init__(self, device_index: int = 0, tag: str = "gpu0"):
        super().__init__()
        self.device_index = device_index
        self.tag = tag

        # Python logger（与你 main.py 用的是同一个 root logger）
        self.py_logger = logging.getLogger(__name__)

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_mem_mb = mem_info.total / 1024**2
            self.py_logger.info(
                f"[GpuMemoryLogger] Total memory on GPU{self.device_index}: "
                f"{self.total_mem_mb:.0f} MB"
            )
        except Exception as e:
            self.handle = None
            self.total_mem_mb = 0
            self.py_logger.warning(f"[GpuMemoryLogger] NVML init failed: {e}")

    def _log_mem(self, trainer, used_mem_mb: float):
        """同时写到 Lightning logger 和 Python logger"""
        ratio = used_mem_mb / self.total_mem_mb if self.total_mem_mb else 0
        pct = ratio * 100

        # 1️⃣ Lightning 日志
        trainer.logger.log_metrics(
            {
                f"{self.tag}_mem_used_mb": used_mem_mb,
                f"{self.tag}_mem_used_pct": pct,
            },
            step=trainer.fit_loop.epoch_progress.current.completed,
        )

        # 2️⃣ Python 日志
        self.py_logger.info(
            f"[GpuMemoryLogger] Epoch {trainer.current_epoch:03d} | "
            f"{self.tag}_mem_used: {used_mem_mb:.1f} MB "
            f"({pct:.1f} %)"
        )

    # ────────────────────────────────────────────────────────────────
    # 你可以改成 on_train_batch_end 以更高频率采样
    # ────────────────────────────────────────────────────────────────
    def on_train_batch_end(self, trainer, pl_module):
        if self.handle is None:
            return
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        used_mem_mb = mem_info.used / 1024**2
        self._log_mem(trainer, used_mem_mb)
