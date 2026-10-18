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


def leer_scn_biorad(ruta_scn: str) -> np.ndarray:
    """
    Lee un .scn de Bio-Rad Image Lab y devuelve la imagen en uint16 con
    fondo CLARO y spots OSCUROS (formato que espera el motor ARY009).

    Conserva los 16 bits originales: no se pierde precisión de intensidad.
    Lanza ValueError si el archivo no es un .scn de Bio-Rad válido.
    """
    with open(ruta_scn, 'rb') as f:
        data = f.read()
    if b'Image Lab' not in data[:300]:
        raise ValueError(f"No parece formato Bio-Rad Image Lab: {ruta_scn}")
    pos = data.find(b'Content-Type: application/octet-stream')
    if pos < 0:
        raise ValueError(f"No se encontró bloque de imagen: {ruta_scn}")
    data_start = data.find(b'\r\n\r\n', pos) + 4
    boundary_m = re.search(rb'boundary="([^"]+)"', data[:pos])
    if boundary_m:
        fin = data.find(b'--' + boundary_m.group(1), data_start + 100)
        if fin < 0:
            fin = len(data)
    else:
        fin = len(data)
    img_data = data[data_start:fin].rstrip(b'\r\n-')
    m = (re.search(rb'size_pix\s+width="(\d+)"\s+height="(\d+)"', data[fin:fin + 10000])
         or re.search(rb'size_pix\s+width="(\d+)"\s+height="(\d+)"', data))
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    else:
        print(f"  ⚠ Sin XML de dimensiones, asumiendo 348x260: {os.path.basename(ruta_scn)}")
        w, h = 348, 260
    dtype = '>u2' if b'<endian>big</endian>' in data else '<u2'
    expected = w * h * 2
    if len(img_data) < expected:
        raise ValueError(f"Datos insuficientes ({len(img_data)} < {expected}) en {ruta_scn}")
    arr = np.frombuffer(img_data[:expected], dtype=dtype).reshape(h, w)
    # zero_is="white": valor alto = oscuro → invertir para fondo claro
    if b'zero_is="white"' in data:
        arr = 65535 - arr
    return arr.astype(np.uint16)


def convertir_scn(ruta_scn: str, ruta_salida: str = None) -> bool:
    """
    Convierte un archivo .scn de Bio-Rad ChemiDoc a PNG de 16 bits.
    (Antes se guardaba en 8 bits y se perdía precisión de intensidad.)
    Returns:
        True si la conversión fue exitosa.
    """
    if not os.path.exists(ruta_scn):
        print(f"  ✗ No existe: {ruta_scn}")
        return False
    try:
        arr16 = leer_scn_biorad(ruta_scn)
    except Exception as e:
        print(f"  ✗ {e}")
        return False
    if ruta_salida is None:
        base, _ = os.path.splitext(ruta_scn)
        ruta_salida = base + '_convertido.png'
    cv2.imwrite(ruta_salida, arr16)  # PNG admite 16 bits sin pérdida
    h, w = arr16.shape
    n_sat = int((arr16 == 0).sum())
    aviso = f"  ⚠ {n_sat} píxeles saturados" if n_sat else ""
    print(f"  ✓ {os.path.basename(ruta_scn):<50} → {os.path.basename(ruta_salida)} ({w}x{h}, 16-bit){aviso}")
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
