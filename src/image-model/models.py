"""Carga de un único pipeline. Importaciones GPU diferidas al proceso hijo."""
import inspect
import time

MODEL_CLASSES = {'generate': 'QwenImagePipeline', 'edit': 'QwenImageEditPlusPipeline'}


def load_model(job, row):
    import torch
    import diffusers
    import transformers
    from huggingface_hub import snapshot_download

    cfg = job['config']
    settings = cfg['models'][job['operation']]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible')
    if settings['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('La GPU no soporta BF16')
    row.update(gpu=torch.cuda.get_device_name(0),
               vram_total_mb=torch.cuda.get_device_properties(0).total_memory / 1048576,
               torch_version=torch.__version__, cuda_version=torch.version.cuda,
               diffusers_version=diffusers.__version__, transformers_version=transformers.__version__)
    # dry_run comprueba TODOS los archivos, sin transferir pesos. Offline nunca consulta Hub.
    start = time.perf_counter()
    kwargs = dict(repo_id=settings['id'], cache_dir=cfg['cache_dir'])
    if job['offline']:
        snapshot = snapshot_download(**kwargs, local_files_only=True)
        row['cache_state'] = 'offline_snapshot_unverified'
    else:
        plan = snapshot_download(**kwargs, dry_run=True)
        if not plan:
            raise RuntimeError('El repositorio no contiene archivos')
        kwargs['revision'] = plan[0].commit_hash
        needed = any(item.will_download for item in plan)
        row['cache_state'] = 'missing_or_partial' if needed else 'complete'
    row['cache_check_seconds'] = time.perf_counter() - start
    row['download_seconds'] = 0.0
    if not job['offline']:
        start = time.perf_counter()
        try:
            snapshot = snapshot_download(**kwargs, local_files_only=not needed)
        finally:
            if needed:
                row['download_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    klass = getattr(diffusers, MODEL_CLASSES[job['operation']])
    pipe = klass.from_pretrained(snapshot, torch_dtype=getattr(torch, settings['dtype']),
                                 local_files_only=True)
    if cfg['offload'] == 'cpu':
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    torch.cuda.synchronize()
    row['load_seconds'] = time.perf_counter() - start
    return pipe


def check_parameters(pipe, kwargs):
    unsupported = set(kwargs) - set(inspect.signature(pipe.__call__).parameters)
    if unsupported:
        raise ValueError(f'Parámetros incompatibles: {sorted(unsupported)}')
