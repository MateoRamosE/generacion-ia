"""Historial de ejecuciones, reloj por fases y muestreo de VRAM."""
import csv
import fcntl
import io
import json
import os
import sys
import tempfile
import threading
import time
from datetime import date

STAGES = (
    'initialization', 'cache_check', 'download', 'model_read', 'model_prepare',
    'inference', 'save', 'cleanup', 'metrics_write',
)
RUN_FIELDS = [
    'date', 'run_id', 'operation', 'model_id', 'status', 'error',
    'gpu', 'vram_total_mb', 'vram_peak_device_used_mb', 'ram_peak_rss_mb',
    *(f'{stage}_seconds' for stage in STAGES), 'total_seconds',
    'downloaded_gb', 'model_size_gb', 'prompts_count', 'images_count',
    'width', 'height', 'steps', 'true_cfg_scale', 'dtype', 'seed', 'offload',
    'gpu_usd_per_hour', 'total_cost_usd',
]


class StageClock:
    def __init__(self, start_ns=None):
        self.start_ns = time.perf_counter_ns() if start_ns is None else start_ns
        self.boundary = self.start_ns
        self.stage = 'initialization'
        self.elapsed = dict.fromkeys(STAGES, 0)

    @classmethod
    def restore(cls, start_ns, boundary, stage, elapsed):
        clock = cls(start_ns)
        clock.boundary = boundary
        clock.stage = stage
        clock.elapsed = elapsed
        return clock

    def switch(self, stage):
        if stage == self.stage:
            return
        now = time.perf_counter_ns()
        self.elapsed[self.stage] += now - self.boundary
        self.boundary = now
        self.stage = stage

    def finish(self):
        self.switch(None)
        return {f'{stage}_seconds': seconds(ns) for stage, ns in self.elapsed.items()}


def seconds(ns):
    whole, fraction = divmod(ns, 10**9)
    return f'{whole}.{fraction:09d}'


def normalize_date(value):
    if not value:
        return ''
    day = value[:10]
    try:
        return date.fromisoformat(day).strftime('%y-%m-%d')
    except ValueError:
        try:
            return date.fromisoformat('20' + day).strftime('%y-%m-%d')
        except ValueError:
            return ''


def historical_row(source):
    row = {key: source.get(key, '') for key in RUN_FIELDS}
    row['date'] = normalize_date(source.get('date') or source.get('started_at')
                                 or source.get('finished_at'))
    try:
        config = json.loads(source.get('parameters_json') or '{}')
        model = config.get('models', {}).get(row['operation'], {})
        for key in ('width', 'height', 'steps', 'true_cfg_scale', 'dtype'):
            if row[key] == '':
                row[key] = model.get(key, '')
        for key in ('offload', 'gpu_usd_per_hour'):
            if row[key] == '':
                row[key] = config.get(key, '')
    except (ValueError, AttributeError):
        pass
    row['offload'] = {'cpu': 'on', 'gpu': 'off'}.get(row['offload'], row['offload'])
    return row


def read_csv(path):
    if path.exists():
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames and 'run_id' not in reader.fieldnames:
                raise ValueError(f'El CSV histórico carece de run_id: {path}')
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'Fila incompleta o incompatible: {path}')
                yield row


def migrate(path, old_runs, old_images):
    runs = {}
    for source_path in (path, old_runs):
        for source in read_csv(source_path):
            row = historical_row(source)
            run_id = row['run_id']
            if not run_id:
                raise ValueError(f'Fila histórica sin run_id: {source_path}')
            saved = runs.setdefault(run_id, row)
            for field, value in row.items():
                if saved[field] in ('', None):
                    saved[field] = value

    images = {}
    for source in read_csv(old_images):
        run_id = source['run_id']
        if not run_id:
            raise ValueError(f'Imagen histórica sin run_id: {old_images}')
        identity = source.get('image_path') or (source.get('prompt_index'), source.get('image_index'))
        images.setdefault(run_id, {})[identity] = source
    for run_id, entries in images.items():
        saved = runs.setdefault(run_id, {'run_id': run_id})
        if saved.get('images_count') in ('', None):
            saved['images_count'] = len(entries)
        for field in ('width', 'height'):
            values = {source.get(field) for source in entries.values()}
            if len(values) == 1 and saved.get(field) in ('', None):
                saved[field] = values.pop()
    return runs


def csv_line(values):
    stream = io.StringIO(newline='')
    csv.writer(stream).writerow(values)
    return stream.getvalue().encode('utf-8')


def write_execution(path, row=None, clock=None):
    """Publica el historial completo de forma atómica bajo bloqueo del directorio."""
    if clock is not None:
        clock.switch('metrics_write')
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        old_runs = path.with_name('executions-v2.csv')
        old_images = path.with_name('images.csv')
        runs = migrate(path, old_runs, old_images)
        if row is None or row['run_id'] not in runs:
            deferred = ('metrics_write_seconds', 'total_seconds', 'total_cost_usd')
            # Reservar espacio para los valores que dependen del fin de la escritura.
            markers = {field: f'@{field:@<39}' for field in deferred}
            with tempfile.NamedTemporaryFile(mode='w+b', dir=path.parent, delete=False) as output:
                temporary = output.name
                try:
                    output.write(csv_line(RUN_FIELDS))
                    for saved in runs.values():
                        output.write(csv_line([saved.get(field, '') for field in RUN_FIELDS]))
                    if row is not None:
                        for stage in STAGES[:-1]:
                            row[f'{stage}_seconds'] = seconds(clock.elapsed[stage])
                        line = csv_line([markers.get(field, row.get(field, '')) for field in RUN_FIELDS])
                        offset = output.tell()
                        output.write(line)
                        output.flush()
                        os.fsync(output.fileno())
                        row.update(clock.finish())
                        row['total_seconds'] = seconds(clock.boundary - clock.start_ns)
                        row['total_cost_usd'] = float(row['total_seconds']) / 3600 * row['gpu_usd_per_hour']
                        for field in deferred:
                            marker = markers[field].encode()
                            output.seek(offset + line.rindex(marker))
                            output.write(str(row[field]).rjust(len(marker)).encode())
                    output.flush()
                    os.fsync(output.fileno())
                    os.replace(temporary, path)
                    os.fsync(directory)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        old_runs.unlink(missing_ok=True)
        old_images.unlink(missing_ok=True)
        os.fsync(directory)
    finally:
        os.close(directory)


class Sampler:
    def __init__(self, interval):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.nvml = self.handle = None
        self.stats = {}

    def add(self, name, value):
        self.stats[name] = max(self.stats.get(name, 0), value)

    def sample(self):
        readers = {}
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
        from resource import RUSAGE_SELF, getrusage

        self.add('ram_rss', getrusage(RUSAGE_SELF).ru_maxrss / 1024)
        row = {'ram_peak_rss_mb': self.stats['ram_rss']}
        if 'vram_device_used' in self.stats:
            row['vram_peak_device_used_mb'] = self.stats['vram_device_used']
        return row
