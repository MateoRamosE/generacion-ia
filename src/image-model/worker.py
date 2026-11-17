"""Coordinador: un modelo por proceso; al salir se destruye el contexto CUDA."""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

from config import validate
from inference import infer
from metrics import RUN_FIELDS, Sampler, append_csv, utcnow
from models import load_model


def execute(job):
    cfg = validate(job['config'])
    settings = cfg['models'][job['operation']]
    start = time.perf_counter()
    row = dict(run_id=job['run_id'], started_at=utcnow(), operation=job['operation'],
               model_id=settings['id'], status='error', prompts_count=len(job['prompts']),
               images_count=0, output_files='[]', seed=job['seed'],
               gpu_usd_per_hour=cfg['gpu_usd_per_hour'], **{
                   k: settings[k] for k in ('width', 'height', 'steps', 'dtype', 'true_cfg_scale')})
    row['parameters_json'] = json.dumps(cfg, ensure_ascii=False)
    sampler = Sampler(cfg['sample_interval_seconds'])
    pipe = None
    code = 0
    try:
        sampler.start()
        pipe = load_model(job, row)
        paths = infer(pipe, job, row)
        row['status'] = 'success'
        print('Imágenes guardadas:', *paths, sep='\n')
    except BaseException as error:
        row['error'] = f'{type(error).__name__}: {error}'
        # No conservar el traceback: puede retener pipeline y tensores.
        traceback.print_exception(type(error), error, error.__traceback__)
        code = 1
    finally:
        try:
            import torch
            if torch.cuda.is_initialized():
                row.update(vram_peak_allocated_mb=torch.cuda.max_memory_allocated() / 1048576,
                           vram_peak_reserved_mb=torch.cuda.max_memory_reserved() / 1048576)
        except ImportError:
            pass
        pipe = None
        gc.collect()
        row.update(sampler.finish())
        row.update(finished_at=utcnow(), total_seconds=time.perf_counter() - start)
        rate = cfg['gpu_usd_per_hour'] / 3600
        row['inference_cost_usd'] = rate * row.get('inference_seconds', 0)
        row['total_cost_usd'] = rate * row['total_seconds']
        append_csv(Path(cfg['metrics_dir']) / 'executions-v2.csv', RUN_FIELDS, row)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(execute(json.loads(args.job.read_text(encoding='utf-8'))))


if __name__ == '__main__':
    main()
