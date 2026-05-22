"""
convertir_scn.py — Convierte archivos .scn de Bio-Rad ChemiDoc a PNG

Los archivos .scn de Bio-Rad están en un formato MIME multipart con datos
binarios uint16 little-endian y atributo zero_is="white" (invertido).

Este script extrae los datos correctamente y genera un PNG en formato
estándar (fondo blanco, spots oscuros) que es compatible con el motor
de análisis ARY009 del programa.

USO:
    python convertir_scn.py archivo.scn [archivo2.scn ...]
    python convertir_scn.py fotos/*.scn
    python convertir_scn.py --carpeta fotos/

Genera para cada entrada un archivo PNG en la misma carpeta con sufijo
"_convertido.png". El archivo original NO se modifica.
"""
import os
import re
import sys
import glob
import argparse

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: Instalá opencv-python y numpy:")
    print("  pip install opencv-python numpy")
    sys.exit(1)


def convertir_scn(ruta_scn: str, ruta_salida: str = None) -> bool:
    """
    Convierte un archivo .scn de Bio-Rad ChemiDoc a PNG estándar.

    Returns:
        True si la conversión fue exitosa.
    """
    if not os.path.exists(ruta_scn):
        print(f"  ✗ No existe: {ruta_scn}")
        return False

    try:
        with open(ruta_scn, 'rb') as f:
            data = f.read()
    except Exception as e:
        print(f"  ✗ Error leyendo {ruta_scn}: {e}")
        return False

    # Verificar formato Bio-Rad
    if b'Image Lab' not in data[:300]:
        print(f"  ✗ No parece formato Bio-Rad Image Lab: {ruta_scn}")
        return False

    # Buscar bloque binario (octet-stream)
    pos = data.find(b'Content-Type: application/octet-stream')
    if pos < 0:
        print(f"  ✗ No se encontró bloque de imagen: {ruta_scn}")
        return False

    data_start = data.find(b'\r\n\r\n', pos) + 4
    # Buscar el siguiente boundary
    boundary_m = re.search(rb'boundary="([^"]+)"', data[:pos])
    if boundary_m:
        boundary = b'--' + boundary_m.group(1)
        fin = data.find(boundary, data_start + 100)
    else:
        fin = len(data)
    img_data = data[data_start:fin].rstrip(b'\r\n-')

    # Buscar dimensiones en el XML que sigue al bloque binario
    xml_zone = data[fin:fin + 10000]
    m = re.search(rb'size_pix\s+width="(\d+)"\s+height="(\d+)"', xml_zone)
    if not m:
        # Buscar en cualquier parte del archivo
        m = re.search(rb'size_pix\s+width="(\d+)"\s+height="(\d+)"', data)

    if m:
        w, h = int(m.group(1)), int(m.group(2))
    else:
        # Fallback: dimensiones típicas
        print(f"  ⚠ Sin XML, asumiendo 348x260")
        w, h = 348, 260

    # Detectar bit depth (zero_is_white indica formato Bio-Rad estándar)
    endian = 'little'
    if b'<endian>big</endian>' in data:
        endian = 'big'
    invertir = b'zero_is="white"' in data or b'<image zero_is="white"/>' in data

    dtype = '<u2' if endian == 'little' else '>u2'
    expected = w * h * 2

    if len(img_data) < expected:
        print(f"  ✗ Datos insuficientes ({len(img_data)} < {expected}) en {ruta_scn}")
        return False

    # Cargar como uint16
    arr = np.frombuffer(img_data[:expected], dtype=dtype).reshape(h, w)

    # Invertir si zero_is_white (alto valor = oscuro)
    if invertir:
        arr_inv = 65535 - arr
    else:
        arr_inv = arr

    # Convertir a 8-bit
    arr8 = (arr_inv / 256).astype(np.uint8)

    # Generar ruta de salida
    if ruta_salida is None:
        base, _ = os.path.splitext(ruta_scn)
        ruta_salida = base + '_convertido.png'

    cv2.imwrite(ruta_salida, arr8)
    print(f"  ✓ {os.path.basename(ruta_scn):<50} → {os.path.basename(ruta_salida)} ({w}x{h})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convierte archivos .scn de Bio-Rad ChemiDoc a PNG"
    )
    parser.add_argument('archivos', nargs='*', help='Archivos .scn a convertir')
    parser.add_argument('--carpeta', help='Convertir todos los .scn de una carpeta')
    args = parser.parse_args()

    rutas = []
    if args.carpeta:
        rutas.extend(sorted(glob.glob(os.path.join(args.carpeta, '*.scn'))))
    rutas.extend(args.archivos)

    if not rutas:
        print("Sin archivos para convertir.")
        print(__doc__)
        return

    print(f"Convirtiendo {len(rutas)} archivo(s):")
    print()
    ok = 0
    for r in rutas:
        if convertir_scn(r):
            ok += 1
    print()
    print(f"Listo: {ok}/{len(rutas)} convertidos exitosamente.")
    print()
    print("Los archivos *_convertido.png pueden cargarse directamente")
    print("en el Analizador ARY009 sin necesidad de procesamiento adicional.")


if __name__ == "__main__":
    main()
