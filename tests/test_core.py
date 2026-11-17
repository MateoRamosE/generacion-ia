import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'image-model'))
import config
import generate
import inference
import metrics
import models
import worker


class CoreTests(unittest.TestCase):
    def test_partial_configuration_and_no_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            self.assertEqual(config.load(path), config.DEFAULT)
            self.assertFalse(path.exists())
            path.write_text('{"models":{"generate":{"steps":12}}}')
            cfg = config.load(path)
            self.assertEqual(cfg['models']['generate']['steps'], 12)
            self.assertEqual(cfg['models']['edit'], config.DEFAULT['models']['edit'])
            config.save(cfg, path)
            self.assertEqual(config.load(path), cfg)

    def test_cli_precedence(self):
        with patch.object(generate, 'load', return_value=copy.deepcopy(config.DEFAULT)):
            cfg = generate.resolve(SimpleNamespace(operation='generate', width=1536))
        self.assertEqual(cfg['models']['generate']['width'], 1536)
        self.assertEqual(config.DEFAULT['models']['generate']['width'], 1024)

    def test_validation(self):
        for key, value in [('batch_size', 0), ('max_images_per_call', True),
                           ('gpu_usd_per_hour', float('nan')), ('offload', 'invalid')]:
            cfg = copy.deepcopy(config.DEFAULT)
            cfg[key] = value
            with self.assertRaises(ValueError):
                config.validate(cfg)
        with self.assertRaises(ValueError):
            config.merge(config.DEFAULT, {'typo': 1})

    def test_batches_cover_once_and_respect_both_limits(self):
        for operation in ('generate', 'edit'):
            for prompts in (1, 3, 5):
                for limit in (1, 2, 4):
                    seen = []
                    for indices, first, count in inference.batches(prompts, 10, 3, limit, operation):
                        self.assertLessEqual(len(indices) * count, limit)
                        self.assertLessEqual(len(indices), 1 if operation == 'edit' else 3)
                        seen.extend((i, j) for i in indices for j in range(first, first + count))
                    self.assertEqual(sorted(seen), [(i, j) for i in range(prompts) for j in range(10)])
        self.assertEqual(len(list(inference.batches(1, 10, 1, 2, 'generate'))), 5)

    def test_names_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {inference.image_path(tmp, 'generate', run, p, i)
                     for run in ('a', 'b') for p in range(2) for i in range(10)}
            self.assertEqual(len(paths), 40)
            path = paths.pop()
            path.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                path.open('xb')
            self.assertEqual(path.read_bytes(), b'original')

    def test_csv_accumulation_and_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'metrics.csv'
            for i in range(3):
                metrics.append_csv(path, ['id', 'text'], {'id': i, 'text': 'a,b\nc'})
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['text'], 'a,b\nc')
            with self.assertRaises(ValueError):
                metrics.append_csv(path, ['different'], {})

    def test_selection_and_signature(self):
        self.assertEqual(models.MODEL_CLASSES['edit'], 'QwenImageEditPlusPipeline')
        class Fake:
            def __call__(self, prompt):
                pass
        with self.assertRaises(ValueError):
            models.check_parameters(Fake(), {'invented': 1})

    def test_process_waits(self):
        child = MagicMock()
        child.wait.return_value = 7
        with patch.object(generate.subprocess, 'Popen', return_value=child):
            self.assertEqual(generate.run_worker({'operation': 'generate'}), 7)
        child.wait.assert_called_once()

    def test_process_interrupt_reaps(self):
        child = MagicMock()
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch.object(generate.subprocess, 'Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                generate.run_worker({})
        child.terminate.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)

    def test_failure_records_once_and_stops_sampler(self):
        cfg = copy.deepcopy(config.DEFAULT)
        with tempfile.TemporaryDirectory() as tmp:
            cfg['metrics_dir'] = tmp
            job = dict(config=cfg, operation='generate', run_id='abc', prompts=['x'], seed=0)
            fake_torch = MagicMock()
            fake_torch.cuda.is_initialized.return_value = False
            with patch.object(worker, 'load_model', side_effect=RuntimeError('load failed')), \
                 patch.object(worker, 'Sampler') as sampler, \
                 patch.dict(sys.modules, {'torch': fake_torch}):
                sampler.return_value.finish.return_value = {}
                self.assertEqual(worker.execute(job), 1)
                sampler.return_value.finish.assert_called_once()
            with (Path(tmp) / 'executions-v2.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertIn('load failed', rows[0]['error'])

    def test_ten_images_five_calls_seeds_and_metrics(self):
        cfg = copy.deepcopy(config.DEFAULT)
        cfg.update(images_per_prompt=10, max_images_per_call=2)
        fake_torch = MagicMock()
        fake_torch.inference_mode.return_value.__enter__.return_value = None
        seeds = []
        fake_torch.Generator.return_value.manual_seed.side_effect = lambda s: seeds.append(s) or s
        class Image:
            width = height = 16
            def save(self, stream, format):
                stream.write(b'mock PNG')
            def close(self):
                pass
        class Pipe:
            calls = 0
            def __call__(self, prompt, num_images_per_prompt, num_inference_steps,
                         true_cfg_scale, negative_prompt, width, height, generator):
                self.calls += 1
                return SimpleNamespace(images=[Image() for _ in range(num_images_per_prompt)])
        pipe = Pipe()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.update(output_dir=tmp, metrics_dir=tmp)
            job = dict(config=cfg, operation='generate', run_id='unique', prompts=['x'],
                       references=[], seed=100)
            row = {}
            with patch.dict(sys.modules, {'torch': fake_torch}):
                paths = inference.infer(pipe, job, row)
            self.assertEqual(len(paths), 10)
            self.assertEqual(pipe.calls, 5)
            self.assertEqual(seeds, list(range(100, 110)))
            self.assertEqual(row['images_count'], 10)
            with (Path(tmp) / 'images.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)


class ModelTests(unittest.TestCase):
    def test_online_cache_complete_and_missing(self):
        for needed in (False, True):
            torch = MagicMock(__version__='mock')
            torch.cuda.get_device_properties.return_value.total_memory = 1024
            diffusers = MagicMock(__version__='mock')
            transformers = SimpleNamespace(__version__='mock')
            hub = MagicMock()
            hub.snapshot_download.side_effect = [
                [SimpleNamespace(will_download=needed, commit_hash='abc')], '/snapshot']
            cfg = copy.deepcopy(config.DEFAULT)
            job = dict(config=cfg, operation='edit', offline=False)
            row = {}
            with patch.dict(sys.modules, dict(torch=torch, diffusers=diffusers,
                                             transformers=transformers, huggingface_hub=hub)):
                pipe = models.load_model(job, row)
            diffusers.QwenImageEditPlusPipeline.from_pretrained.assert_called_once()
            pipe.enable_model_cpu_offload.assert_called_once()
            self.assertEqual(hub.snapshot_download.call_args.kwargs['local_files_only'], not needed)
            self.assertEqual(hub.snapshot_download.call_args.kwargs['revision'], 'abc')
            self.assertEqual(row['cache_state'], 'missing_or_partial' if needed else 'complete')
            if not needed:
                self.assertEqual(row['download_seconds'], 0)

    def test_offline_never_uses_remote_plan(self):
        torch = MagicMock(__version__='mock')
        torch.cuda.get_device_properties.return_value.total_memory = 1024
        hub = MagicMock()
        hub.snapshot_download.return_value = '/cached'
        cfg = copy.deepcopy(config.DEFAULT)
        with patch.dict(sys.modules, dict(torch=torch, diffusers=MagicMock(__version__='mock'),
                                         transformers=SimpleNamespace(__version__='mock'),
                                         huggingface_hub=hub)):
            models.load_model(dict(config=cfg, operation='generate', offline=True), {})
        self.assertEqual(hub.snapshot_download.call_count, 1)
        self.assertTrue(hub.snapshot_download.call_args.kwargs['local_files_only'])

    def test_sampler_constant_memory_and_finish(self):
        sampler = metrics.Sampler(0.01)
        for i in range(1000):
            sampler.add('ram_rss', i)
        row = sampler.finish()
        self.assertEqual(row['ram_peak_rss_mb'], 999)
        self.assertEqual(row['ram_avg_rss_mb'], 499.5)
        self.assertEqual(len(sampler.stats), 1)

class ResourceTests(unittest.TestCase):
    def test_pipeline_reference_released_after_failure(self):
        import weakref
        class Pipeline:
            pass
        refs = []
        def load(job, row):
            pipe = Pipeline()
            refs.append(weakref.ref(pipe))
            return pipe
        def fail(pipe, job, row):
            raise RuntimeError('inference failed')
        cfg = copy.deepcopy(config.DEFAULT)
        fake_torch = MagicMock()
        fake_torch.cuda.is_initialized.return_value = False
        with tempfile.TemporaryDirectory() as tmp:
            cfg['metrics_dir'] = tmp
            with patch.object(worker, 'load_model', side_effect=load), \
                 patch.object(worker, 'infer', side_effect=fail), \
                 patch.object(worker, 'Sampler') as sampler, \
                 patch.object(worker.traceback, 'print_exception'), \
                 patch.dict(sys.modules, {'torch': fake_torch}):
                sampler.return_value.finish.return_value = {}
                self.assertEqual(worker.execute(dict(config=cfg, operation='generate',
                                                     run_id='release', prompts=['x'], seed=None)), 1)
        import gc
        gc.collect()
        self.assertIsNone(refs[0]())

    def test_reference_conversion_closes_sources_and_outputs(self):
        source = MagicMock()
        source.__enter__.return_value = source
        rgb = source.convert.return_value
        pil = MagicMock()
        pil.Image.open.return_value = source
        with patch.dict(sys.modules, {'PIL': pil}):
            refs = inference.read_references(['one', 'two'])
        self.assertEqual(len(refs), 2)
        self.assertEqual(source.__exit__.call_count, 2)
        self.assertIs(refs[0], rgb)

if __name__ == '__main__':
    unittest.main()
