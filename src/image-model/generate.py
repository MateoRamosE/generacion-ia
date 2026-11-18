"""CLI sin dependencias GPU; cada trabajo espera a un proceso aislado."""
import time

START_NS = time.perf_counter_ns() if __name__ == "__main__" else None

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timezone

from config import CONFIG, load, reset, save, save_overrides
from metrics import StageClock, write_execution

GLOBAL = {"output_dir": str, "metrics_dir": str, "cache_dir": str,
          "gpu_usd_per_hour": float, "sample_interval_seconds": float,
          "batch_size": int, "max_images_per_call": int, "images_per_prompt": int,
          "offload": str}
MODEL = {"width": int, "height": int, "steps": int, "true_cfg_scale": float,
         "dtype": str, "negative_prompt": str}


def add_settings(parser):
    for key, kind in {**GLOBAL, **MODEL}.items():
        flag = '--' + key.replace('_', '-')
        if key == 'gpu_usd_per_hour':
            parser.add_argument(flag, '--pod-usd', type=kind)
        else:
            parser.add_argument(flag, type=kind)
    parser.add_argument('--max-prompts-per-call', dest='batch_size', type=int)


def resolve(args):
    from config import validate
    cfg = load()
    for key in GLOBAL:
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    operation = getattr(args, 'model', None) or args.operation
    for key in MODEL:
        if getattr(args, key, None) is not None:
            cfg['models'][operation][key] = getattr(args, key)
    return validate(cfg)


def run_worker(job):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'job.json'
        path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        report = Path(tmp) / 'report.json'
        child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('worker.py')),
                                  '--job', str(path), '--report', str(report)])
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise

        outcome = json.loads(report.read_text(encoding='utf-8')) if report.exists() else None
    if outcome:
        clock = StageClock.restore(**outcome['clock'])
        write_execution(Path(job['config']['metrics_dir']) / 'executions.csv', outcome['row'], clock)
    return code


def main():
    date = datetime.now(timezone.utc).strftime('%y-%m-%d')
    parser = argparse.ArgumentParser(description='Qwen en RunPod')
    sub = parser.add_subparsers(dest='operation', required=True)
    for op in ('generate', 'edit'):
        q = sub.add_parser(op)
        group = q.add_mutually_exclusive_group(required=True)
        group.add_argument('--prompt')
        group.add_argument('--prompts-file', type=Path)
        q.add_argument('--reference', action='append', default=[])
        q.add_argument('--seed', type=int)
        q.add_argument('--offline', action='store_true')
        add_settings(q)
    sub.add_parser('config-show')
    migrate = sub.add_parser('metrics-migrate')
    migrate.add_argument('--metrics-dir', type=Path)
    sub.add_parser('config-reset')
    q = sub.add_parser('config-set')
    q.add_argument('model', choices=('generate', 'edit'))
    add_settings(q)
    args = parser.parse_args()
    try:
        if args.operation == 'metrics-migrate':
            directory = args.metrics_dir or Path(load()['metrics_dir'])
            write_execution(directory / 'executions.csv')
            print(directory / 'executions.csv')
            return
        if args.operation == 'config-show':
            print(json.dumps(load(), indent=2, ensure_ascii=False))
            return
        if args.operation == 'config-reset':
            reset()
            print('Configuración restablecida:', CONFIG)
            return
        cfg = resolve(args)
        if args.operation == 'config-set':
            save(cfg)
            return
        prompts = [args.prompt] if args.prompt is not None else [
            line.strip() for line in args.prompts_file.read_text(encoding='utf-8').splitlines()
            if line.strip()]
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError('Se requiere un prompt no vacío')
        if bool(args.reference) != (args.operation == 'edit'):
            raise ValueError('edit requiere referencias; generate no las admite')
        if args.seed is not None and not 0 <= args.seed < 2**63:
            raise ValueError('seed debe estar entre 0 y 2**63-1')
        for path in args.reference:
            if not Path(path).is_file():
                raise ValueError(f'No existe la referencia: {path}')
        persistent = {key: cfg[key] for key in ('gpu_usd_per_hour', 'offload')
                      if getattr(args, key) is not None}
        if persistent:
            save_overrides(persistent)
        job = dict(start_ns=START_NS, date=date, run_id=uuid.uuid4().hex, operation=args.operation, prompts=prompts,
                   references=[str(Path(x).resolve()) for x in args.reference],
                   seed=args.seed, offline=args.offline, config=cfg)
        code = run_worker(job)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    else:
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
