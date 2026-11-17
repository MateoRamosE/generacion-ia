"""CSV acumulativos y muestreo de memoria con almacenamiento constante."""
import csv
import fcntl
import os
import sys
import threading
from datetime import datetime, timezone

RUN_FIELDS = ['run_id', 'started_at', 'finished_at', 'operation', 'model_id', 'status',
              'error', 'gpu', 'vram_total_mb', 'vram_peak_allocated_mb',
              'vram_peak_reserved_mb', 'vram_avg_allocated_mb', 'vram_avg_reserved_mb',
              'vram_peak_device_used_mb', 'vram_avg_device_used_mb', 'ram_peak_rss_mb',
              'ram_avg_rss_mb', 'cache_check_seconds', 'download_seconds', 'load_seconds',
              'inference_seconds', 'save_seconds', 'total_seconds', 'cache_state',
              'prompts_count', 'images_count', 'width', 'height', 'steps', 'true_cfg_scale',
              'dtype', 'seed', 'torch_version', 'cuda_version', 'diffusers_version',
              'transformers_version', 'gpu_usd_per_hour', 'inference_cost_usd',
              'total_cost_usd', 'output_files', 'parameters_json']
IMAGE_FIELDS = ['run_id', 'image_path', 'prompt_index', 'image_index', 'width', 'height', 'seed']


def append_csv(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', newline='', encoding='utf-8') as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.seek(0)
        header = next(csv.reader(output), None)
        if header is not None and header != fields:
            raise ValueError(f'CSV incompatible: {path}')
        output.seek(0, 2)
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        if header is None:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fields})


class Sampler:
    def __init__(self, interval):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.process = self.nvml = self.handle = None
        self.stats = {}

    def add(self, name, value):
        count, total, peak = self.stats.get(name, (0, 0, 0))
        self.stats[name] = count + 1, total + value, max(peak, value)

    def sample(self):
        readers = {}
        if self.process:
            readers['ram_rss'] = lambda: self.process.memory_info().rss
        if self.nvml and self.handle is None:
            try:
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
                if visible.startswith(('GPU-', 'MIG-')):
                    self.handle = self.nvml.nvmlDeviceGetHandleByUUID(visible)
                else:
                    torch = sys.modules.get('torch')
                    if torch and torch.cuda.is_initialized():
                        # CUDA logical 0 may map to a different NVML physical index.
                        uuid = str(torch.cuda.get_device_properties(0).uuid)
                        self.handle = self.nvml.nvmlDeviceGetHandleByUUID(uuid)
            except Exception:
                pass
        if self.handle is not None:
            readers['vram_device_used'] = lambda: self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        torch = sys.modules.get('torch')
        if torch and torch.cuda.is_initialized():
            readers['vram_allocated'] = torch.cuda.memory_allocated
            readers['vram_reserved'] = torch.cuda.memory_reserved
        for name, read in readers.items():
            try:
                self.add(name, read() / 1048576)
            except Exception:
                # NVML puede no estar disponible; dejar celdas vacías, no ceros ficticios.
                continue

    def loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def start(self):
        try:
            import psutil
            self.process = psutil.Process(os.getpid())
        except ImportError:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
        except Exception:
            pass
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.sample()
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        row = {}
        for name, (count, total, peak) in self.stats.items():
            family, suffix = name.split('_', 1)
            row[f'{family}_avg_{suffix}_mb'] = total / count
            if name not in ('vram_allocated', 'vram_reserved'):
                row[f'{family}_peak_{suffix}_mb'] = peak
        return row


def utcnow():
    return datetime.now(timezone.utc).isoformat()
