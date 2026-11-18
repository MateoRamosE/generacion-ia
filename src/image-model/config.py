import copy
import fcntl
import json
import math
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.json"
DEFAULT = {
 "cache_dir": "/workspace/modelos/huggingface/hub",
 "output_dir": "/workspace/generacion-ia/results/images",
 "gpu_usd_per_hour": 0.0,
 "sample_interval_seconds": 0.5,
 "batch_size": 1,
 "max_images_per_call": 1,
 "images_per_prompt": 1,
 "metrics_dir": str(ROOT / "results" / "metrics"),
 "offload": "on",
 "models": {
  "generate": {"id": "Qwen/Qwen-Image-2512", "dtype": "bfloat16", "width": 1024, "height": 1024, "steps": 50, "true_cfg_scale": 1.0, "negative_prompt": " "},
  "edit": {"id": "Qwen/Qwen-Image-Edit-2511", "dtype": "bfloat16", "width": None, "height": None, "steps": 40, "true_cfg_scale": 4.0, "negative_prompt": " "}
 }
}


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Parámetro desconocido: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} debe ser un objeto")
            result[key] = merge(base[key], value)
        else:
            result[key] = value
    return result


def validate(cfg):
    for key in ("batch_size", "max_images_per_call", "images_per_prompt"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} debe ser entero >= 1")
    for key in ("gpu_usd_per_hour", "sample_interval_seconds"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} debe ser un número finito no negativo")
    if cfg["sample_interval_seconds"] == 0:
        raise ValueError("El intervalo debe ser positivo")
    offload = cfg["offload"]
    if offload in ("cpu", "gpu"):
        cfg["offload"] = "on" if offload == "cpu" else "off"
    elif offload not in ("on", "off"):
        raise ValueError("offload debe ser on u off")
    for key in ("cache_dir", "output_dir", "metrics_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} debe ser una ruta")
    for op, model in cfg["models"].items():
        if model["id"] != DEFAULT["models"][op]["id"]:
            raise ValueError("No se permite sustituir el modelo implícitamente")
        if model["dtype"] not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype inválido")
        for key in ("width", "height", "steps"):
            value = model[key]
            if value is None and key != "steps":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} debe ser entero positivo")
            if key in ("width", "height") and value % 16:
                raise ValueError(f"{key} debe ser múltiplo de 16")
        scale = model["true_cfg_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale < 1:
            raise ValueError("true_cfg_scale debe ser finito >= 1")
        if not isinstance(model["negative_prompt"], str):
            raise ValueError("negative_prompt debe ser texto")
    return cfg


def load(path=CONFIG):
    updates = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return validate(merge(DEFAULT, updates))


def save(cfg, path=CONFIG):
    validate(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reset():
    save(copy.deepcopy(DEFAULT))


def save_overrides(updates, path=CONFIG):
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        save({**load(path), **updates}, path)
    finally:
        os.close(directory)
