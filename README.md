# Generación y edición Qwen en RunPod

Proyecto refactorizado para Python **3.11 o superior**, PyTorch **2.6 o superior** con CUDA y GPU NVIDIA. BF16 es el perfil predeterminado y requiere soporte de hardware. No se han descargado pesos ni ejecutado inferencia real durante esta entrega.

## Instalación

Descomprime el proyecto en `/workspace/generacion-ia`. Al actualizar una instalación, conserva su `config.json` y todo `results/`; no reemplaces tu configuración con la del ZIP.

Usa la instalación PyTorch CUDA de tu imagen RunPod. Compruébala antes de instalar:

```bash
cd /workspace/generacion-ia
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip install -r requirements.txt
python -m pip check
python src/image-model/generate.py config-show
```

Las versiones estables seleccionadas y verificadas en PyPI son Diffusers 0.43.0, Transformers 5.23.0 y Hugging Face Hub 2.4.0. Las dependencias transitivas no están bloqueadas completamente. Si PyTorch falta, Accelerate puede instalar una distribución CUDA diferente; prepara primero la imagen apropiada del Pod. No se valida aquí esa combinación con tu controlador.

## Uso

```bash
python src/image-model/generate.py generate --prompt "Un automóvil deportivo" --width 1536 --height 1024 --steps 40
python src/image-model/generate.py generate --prompts-file prompts.txt --images-per-prompt 10 --max-images-per-call 2 --max-prompts-per-call 1 --seed 123
python src/image-model/generate.py edit --prompt "Cambia el cielo al atardecer" --reference a.png
python src/image-model/generate.py edit --prompt "Combina ambos sujetos" --reference a.png --reference b.png --images-per-prompt 4 --max-images-per-call 1
python src/image-model/generate.py generate --prompt "Un bosque" --offline --output-dir /workspace/mis-imagenes
```

`--images-per-prompt` es el total por prompt. `--max-images-per-call` limita el total de imágenes de **toda una llamada**, y `--max-prompts-per-call` limita los prompts simultáneos (alias histórico: `--batch-size`). El plan respeta ambos límites; puede dejar capacidad libre para no mezclar cantidades diferentes por prompt. Diez imágenes con límite dos y un prompt producen cinco llamadas consecutivas con el mismo pipeline cargado. No hay reintento automático ante OOM: reduce los límites o resolución.

Edición siempre procesa un prompt por llamada: las referencias son una lista de imágenes del mismo trabajo y se reutilizan para todos sus prompts. Se conservan múltiples referencias y múltiples salidas por prompt sin inventar lotes anidados de referencias. El número de referencias consume memoria adicional y queda sujeto a las capacidades del modelo.

Las semillas son `seed + prompt_index * total_images + image_index` módulo `2**63`, una por imagen. No se reinicia la secuencia al cambiar de lote. CUDA y distintos tamaños de lote no garantizan identidad bit a bit.

## Configuración persistente y temporal

Prioridad: terminal > `config.json` > `DEFAULT` de `config.py`. El archivo JSON conserva los ajustes del usuario; Python define valores base, combinación y validación. Leer configuración no escribe archivos. Los campos omitidos heredan defaults; los desconocidos generan error para detectar erratas.

```bash
python src/image-model/generate.py config-show
python src/image-model/generate.py config-set generate --width 1536 --height 1024 --steps 40 --images-per-prompt 10 --max-images-per-call 2
python src/image-model/generate.py config-set edit --steps 40 --true-cfg-scale 4 --negative-prompt "borroso"
python src/image-model/generate.py config-set generate --gpu-usd-per-hour 0.79 --offload cpu
python src/image-model/generate.py config-reset
```

`config-set` guarda explícitamente: parámetros de modelo sólo en el perfil seleccionado, parámetros globales para ambas operaciones. `config-reset` restablece la configuración completa sin tocar resultados. Para devolver una dimensión de edición a `null`, edita explícitamente el JSON o restablece defaults.

También puedes ajustar `--dtype`, `--cache-dir`, `--metrics-dir` y `--sample-interval-seconds`. Ancho y alto deben ser múltiplos de 16 (la rejilla del VAE/patches de Qwen). Edición permite dimensiones `null`, determinadas por el pipeline. Los perfiles y resoluciones no garantizan caber en VRAM.

Se añade `negative_prompt: " "` por defecto porque el CFG verdadero de Qwen sólo se activa cuando existe un prompt negativo. El worker anterior pasaba `true_cfg_scale` sin prompt negativo: esa escala podía quedar desactivada. Usa `--negative-prompt ""` para recuperar ese comportamiento, o escribe un negativo propio.

## Arquitectura

```text
generacion-ia/
├── config.json
├── requirements.txt
├── prompts.txt
├── README.md
├── DIAGNOSTICO.md
├── VERIFICACION.md
├── src/image-model/
│   ├── generate.py    CLI y ciclo de vida del proceso hijo
│   ├── config.py      defaults, combinación, validación y escritura atómica
│   ├── worker.py      coordinación de una ejecución y cierre de métricas
│   ├── models.py      selección, caché, carga y contrato de pipeline
│   ├── inference.py   lotes, semillas, referencias y archivos de salida
│   └── metrics.py     CSV con bloqueo y muestreo acotado de memoria
├── tests/test_core.py
└── results/
    ├── images/
    └── metrics/
```

No hay jerarquía de adaptadores: ambos pipelines comparten casi todos los argumentos; únicamente selección de clase, referencias y límite de prompts necesitan distinción.

## Modelos y recursos

Se mantienen `Qwen/Qwen-Image-2512` → `QwenImagePipeline` y `Qwen/Qwen-Image-Edit-2511` → `QwenImageEditPlusPipeline`. El proceso principal no importa PyTorch. Cada ejecución inicia y espera un único hijo que carga un modelo, lo reutiliza para todas las llamadas y sale. Al interrumpirlo, el padre termina y recoge el hijo; si no termina en 15 segundos, lo mata y espera su salida.

Los pesos persisten en `/workspace/modelos/huggingface/hub`; no son modelos cargados en RAM o VRAM. CPU offload mueve componentes entre RAM y VRAM. Al finalizar se eliminan referencias al pipeline y se recolecta basura; la salida del proceso destruye el contexto CUDA. No se utiliza `empty_cache()` como liberador de modelos. Ejecutar varios CLI independientes en paralelo sí puede cargar varios modelos: ejecuta trabajos secuencialmente en el Pod.

Online se consulta el plan `snapshot_download(dry_run=True)` completo y se fija su commit para la descarga. Se reutilizan archivos presentes, descargando los faltantes. Offline no consulta Hub y usa el snapshot local; no demuestra su integridad: el cargador informará archivos ausentes. No se borra caché ni se reemplazan modelos.

## Almacenamiento y métricas

Imágenes predeterminadas en `/workspace/generacion-ia/results/images`. Nombre: `operacion-run_id-prompt_index-image_index.png`, índices desde cero y run_id UUID. Apertura exclusiva: una colisión produce error en lugar de sobrescribir.

Las métricas predeterminadas siguen en `results/metrics` del repositorio. `images.csv` conserva el esquema histórico. Las ejecuciones nuevas usan **`executions-v2.csv`** porque añadir columnas al CSV anterior corrompería su encabezado. `executions.csv` queda intacto. Hay bloqueo de archivo Linux entre escritores y comprobación estricta de encabezados.

Cada imagen enlaza su run_id, índices y semilla efectiva. El registro de ejecución incluye modelo, operación, parámetros completos en JSON, recuentos, dimensiones reales, versiones, memoria, estado/error, caché, tiempos y costos estimados. Los contadores y tiempos se acumulan antes de terminar: si falla una llamada posterior, se conservan las imágenes ya registradas y su información. No se reintenta automáticamente la escritura de una fila.

- `cache_check_seconds`: planificación online o resolución offline, sin transferencia de pesos.
- `download_seconds`: cero con caché completa; cuando faltan archivos, tiempo de la operación de descarga (incluye coordinación y trabajo local, no tiempo puro de red).
- `load_seconds`: carga y colocación de componentes, con sincronización CUDA.
- `inference_seconds`: llamadas de inferencia sincronizadas con CUDA.
- `save_seconds`: escritura de imágenes; no incluye la escritura de CSV.
- `total_seconds`: ejecución del worker, incluido cierre de muestreo y liberación, excepto el arranque del intérprete y escritura final de su CSV.

RAM = RSS del proceso. PyTorch distingue memoria asignada y reservada: promedios muestreados y máximos del asignador, incluida la carga. NVML mide memoria total del dispositivo, incluidos otros procesos; se busca la GPU por UUID de CUDA o `CUDA_VISIBLE_DEVICES`. Si no puede identificarse o NVML falla, deja campos vacíos. Los promedios son medias de muestras y los picos NVML/RSS pueden omitir ráfagas entre muestras. El muestreo usa almacenamiento constante.

Tarifa manual en USD/h × segundos / 3600; se registran costo de inferencia y costo total. Son estimaciones, no facturación exacta de RunPod. SIGKILL, corte de energía o un disco inaccesible pueden impedir la fila final. Imagen y CSV no son una transacción: un fallo entre ambas escrituras puede dejar un archivo sin fila. No hay recuperación automática ni reanudación de trabajos interrumpidos.

## Pruebas

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

No descargan modelos ni requieren CUDA. Véase `VERIFICACION.md` para los resultados y lo pendiente en RunPod.
