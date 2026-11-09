
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from diffusers import QwenImage21Pipeline


# --------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Comprobar GPU NVIDIA
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "No se detectó una GPU NVIDIA disponible mediante CUDA."
    )

gpu_name = torch.cuda.get_device_name(0)

# Elegir precisión según el soporte de la GPU
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16

print(f"\nGPU detectada: {gpu_name}")
print(f"Precisión utilizada: {dtype}")


# --------------------------------------------------
# Cargar Qwen-Image 2.1
# --------------------------------------------------

print("\nCargando Qwen-Image 2.1...")

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    dtype=dtype
).to("cuda")

print("Modelo cargado correctamente.")


# --------------------------------------------------
# Solicitar prompt
# --------------------------------------------------

prompt = input("\nDescribe la imagen que quieres generar:\n> ").strip()

if not prompt:
    raise ValueError("El prompt no puede estar vacío.")


# --------------------------------------------------
# Solicitar imágenes de referencia
# --------------------------------------------------

while True:
    try:
        cantidad = int(
            input("\n¿Cuántas imágenes de referencia quieres usar? (0-10): ")
        )

        if 0 <= cantidad <= 10:
            break

        print("Debes introducir un número entre 0 y 10.")

    except ValueError:
        print("Introduce un número entero válido.")


referencias = []

for i in range(cantidad):

    while True:
        ruta = input(
            f"\nRuta de la imagen de referencia {i + 1}:\n> "
        ).strip().strip('"')

        ruta = Path(ruta).expanduser()

        # Las rutas relativas se interpretan desde la raíz del proyecto
        if not ruta.is_absolute():
            ruta = ROOT / ruta

        if not ruta.is_file():
            print(f"No se encontró el archivo: {ruta}")
            continue

        try:
            with Image.open(ruta) as imagen:
                referencias.append(imagen.convert("RGB"))

            break

        except Exception as error:
            print(f"No se pudo abrir la imagen: {error}")


# --------------------------------------------------
# Generar imagen
# --------------------------------------------------

print("\nGenerando imagen...")

resultado = pipe(
    prompt=prompt,
    image=referencias if referencias else None
).images[0]


# --------------------------------------------------
# Guardar resultado
# --------------------------------------------------

fecha = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
ruta_salida = OUTPUT_DIR / f"qwen-image-2.1-{fecha}.png"

resultado.save(ruta_salida)

print("\nImagen generada correctamente.")
print(f"Ubicación: {ruta_salida}")
