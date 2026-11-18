"""Planificación acotada, referencias y persistencia de imágenes."""
from pathlib import Path

from metrics import StageClock
from models import check_parameters


def batches(prompts_count, total, max_prompts, max_images, operation):
    # En edit una lista de imágenes representa referencias, no un lote de prompts.
    prompt_limit = 1 if operation == 'edit' else min(max_prompts, max_images)
    for offset in range(0, prompts_count, prompt_limit):
        indices = list(range(offset, min(offset + prompt_limit, prompts_count)))
        chunk = max(1, max_images // len(indices))
        for first in range(0, total, chunk):
            yield indices, first, min(chunk, total - first)


def image_path(directory, operation, run_id, prompt_index, image_index):
    return Path(directory) / f'{operation}-{run_id}-{prompt_index:03d}-{image_index:03d}.png'


def read_references(paths):
    from PIL import Image
    refs = []
    try:
        for path in paths:
            with Image.open(path) as source:
                refs.append(source.convert('RGB'))
        return refs
    except BaseException:
        for ref in refs:
            ref.close()
        raise


def infer(pipe, job, row, clock=None):
    if clock is None:
        clock = StageClock()
    clock.switch('initialization')
    import torch

    cfg = job['config']
    op = job['operation']
    settings = cfg['models'][op]
    directory = Path(cfg['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    refs = read_references(job['references'])
    paths = []
    total = cfg['images_per_prompt']
    dimensions = set()
    try:
        for indices, first, count in batches(len(job['prompts']), total, cfg['batch_size'],
                                            cfg['max_images_per_call'], op):
            clock.switch('inference')
            prompts = [job['prompts'][i] for i in indices]
            kwargs = dict(prompt=prompts if len(prompts) > 1 else prompts[0],
                          num_images_per_prompt=count, num_inference_steps=settings['steps'],
                          true_cfg_scale=settings['true_cfg_scale'],
                          negative_prompt=settings['negative_prompt'])
            for key in ('width', 'height'):
                if settings[key] is not None:
                    kwargs[key] = settings[key]
            if op == 'edit':
                kwargs['image'] = refs if len(refs) > 1 else refs[0]
            if job['seed'] is not None:
                seeds = [(job['seed'] + i * total + j) % 2**63
                         for i in indices for j in range(first, first + count)]
                kwargs['generator'] = [torch.Generator(device='cpu').manual_seed(s) for s in seeds]
            check_parameters(pipe, kwargs)
            torch.cuda.synchronize()
            with torch.inference_mode():
                result = pipe(**kwargs)
            try:
                torch.cuda.synchronize()
                if len(result.images) != len(indices) * count:
                    raise RuntimeError('El pipeline devolvió una cantidad inesperada de imágenes')
                clock.switch('save')
                for n, im in enumerate(result.images):
                    pi, ii = indices[n // count], first + n % count
                    path = image_path(directory, op, job['run_id'], pi, ii)
                    output = path.open('xb')
                    try:
                        with output:
                            im.save(output, format='PNG')
                    except BaseException:
                        path.unlink()
                        raise
                    paths.append(str(path))
                    dimensions.add((im.width, im.height))
                    width, height = next(iter(dimensions)) if len(dimensions) == 1 else ('', '')
                    row.update(images_count=len(paths), width=width, height=height)
            finally:
                clock.switch('cleanup')
                for image in result.images:
                    image.close()
                del result
    finally:
        clock.switch('cleanup')
        for ref in refs:
            ref.close()
    return paths
