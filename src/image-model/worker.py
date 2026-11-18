"""Coordinador: un modelo por proceso; al salir se destruye el contexto CUDA."""
import argparse
import gc
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from config import validate
from inference import infer
from metrics import Sampler, StageClock, write_execution
from models import load_model


def execute(job, report=None):
    clock = StageClock(job.get('start_ns'))
    cfg = validate(job['config'])
    settings = cfg['models'][job['operation']]
    row = dict(run_id=job['run_id'], date=job.get('date') or
               datetime.now(timezone.utc).strftime('%y-%m-%d'),
               operation=job['operation'], model_id=settings['id'], status='error',
               prompts_count=len(job['prompts']), images_count=0, seed=job['seed'],
               offload=cfg['offload'], gpu_usd_per_hour=cfg['gpu_usd_per_hour'], **{
                   k: settings[k] for k in ('width', 'height', 'steps', 'dtype', 'true_cfg_scale')})
    sampler = Sampler(cfg['sample_interval_seconds'])
    pipe = None
    code = 0
    try:
        sampler.start()
        pipe = load_model(job, row, clock)
        paths = infer(pipe, job, row, clock)
        row['status'] = 'success'
        print('Imágenes guardadas:', *paths, sep='\n')
    except BaseException as error:
        clock.switch('cleanup')
        row['error'] = f'{type(error).__name__}: {error}'
        torch = sys.modules.get('torch')
        oom_type = getattr(torch, 'OutOfMemoryError', None)
        cuda_oom = isinstance(oom_type, type) and isinstance(error, oom_type)
        if cuda_oom and cfg['offload'] == 'off':
            row['error'] += ' | Prueba --offload on o reduce la resolución o el tamaño del lote.'
        elif cuda_oom:
            row['error'] += ' | Reduce la resolución o el tamaño del lote.'
        elif isinstance(error, MemoryError):
            row['error'] += ' | Memoria RAM insuficiente.'
        traceback.print_exception(type(error), error, error.__traceback__)
        if cuda_oom or isinstance(error, MemoryError):
            print(row['error'], file=sys.stderr)
        code = 1
    finally:
        clock.switch('cleanup')
        pipe = None
        gc.collect()
        row.update(sampler.finish())
        if report is None:
            write_execution(Path(cfg['metrics_dir']) / 'executions.csv', row, clock)
        else:
            report.write_text(json.dumps({'row': row, 'clock': vars(clock)}), encoding='utf-8')
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    raise SystemExit(execute(json.loads(args.job.read_text(encoding='utf-8')), args.report))


if __name__ == '__main__':
    main()
