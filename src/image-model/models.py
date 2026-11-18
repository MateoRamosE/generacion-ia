"""Resolución de caché, transferencia medida y carga de un único pipeline."""
import inspect
import threading
from contextlib import contextmanager
from pathlib import Path

from metrics import StageClock

MODEL_CLASSES = {'generate': 'QwenImagePipeline', 'edit': 'QwenImageEditPlusPipeline'}
GB = 10**9


def download_snapshot(snapshot_download, kwargs, row):
    from huggingface_hub import constants, file_download

    stream_backoff = file_download.http_stream_backoff
    transferred = 0
    lock = threading.Lock()

    @contextmanager
    def counted_stream(*args, **options):
        nonlocal transferred
        with stream_backoff(*args, **options) as response:
            try:
                yield response
            finally:
                with lock:
                    transferred += response.num_bytes_downloaded

    # Hub 2.4 no expone los bytes de red de Xet a Python. En este worker,
    # los archivos faltantes se transfieren por el backend HTTP del Hub.
    disable_xet = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = True
    file_download.http_stream_backoff = counted_stream
    try:
        return snapshot_download(**kwargs, local_files_only=False)
    finally:
        file_download.http_stream_backoff = stream_backoff
        constants.HF_HUB_DISABLE_XET = disable_xet
        row['downloaded_gb'] = transferred / GB


def snapshot_size(snapshot):
    total = 0
    seen = set()
    for path in Path(snapshot).rglob('*'):
        if path.is_file():
            entry = path.stat()
            identity = (entry.st_dev, entry.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += entry.st_size
    return total / GB


def load_model(job, row, clock=None):
    if clock is None:
        clock = StageClock()
    import torch
    import diffusers
    from huggingface_hub import snapshot_download

    cfg = job['config']
    settings = cfg['models'][job['operation']]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible')
    properties = torch.cuda.get_device_properties(0)
    row.update(gpu=torch.cuda.get_device_name(0),
               vram_total_mb=properties.total_memory / 1048576)
    if settings['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('La GPU no soporta BF16')

    clock.switch('cache_check')
    kwargs = dict(repo_id=settings['id'], cache_dir=cfg['cache_dir'])
    if job['offline']:
        snapshot = snapshot_download(**kwargs, local_files_only=True)
        row['downloaded_gb'] = 0
        cache_message = 'caché local (offline)'
    else:
        plan = snapshot_download(**kwargs, dry_run=True)
        if not plan:
            raise RuntimeError('El repositorio no contiene archivos')
        kwargs['revision'] = plan[0].commit_hash
        missing = sum(item.will_download for item in plan)
        if missing:
            clock.switch('download')
            snapshot = download_snapshot(snapshot_download, kwargs, row)
            clock.switch('cache_check')
            if row['downloaded_gb'] == 0:
                cache_message = 'caché reutilizada'
            else:
                cache_message = 'descarga completa' if missing == len(plan) else 'descarga parcial'
        else:
            snapshot = snapshot_download(**kwargs, local_files_only=True)
            row['downloaded_gb'] = 0
            cache_message = 'caché reutilizada'
    row['model_size_gb'] = snapshot_size(snapshot)
    print('Modelo:', cache_message)

    clock.switch('model_read')
    klass = getattr(diffusers, MODEL_CLASSES[job['operation']])
    pipe = klass.from_pretrained(snapshot, torch_dtype=getattr(torch, settings['dtype']),
                                 local_files_only=True)
    clock.switch('model_prepare')
    if cfg['offload'] == 'on':
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    torch.cuda.synchronize()
    return pipe


def check_parameters(pipe, kwargs):
    unsupported = set(kwargs) - set(inspect.signature(pipe.__call__).parameters)
    if unsupported:
        raise ValueError(f'Parámetros incompatibles: {sorted(unsupported)}')
