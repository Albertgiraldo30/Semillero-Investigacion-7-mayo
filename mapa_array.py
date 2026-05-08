"""
mapa_array.py — Mapa oficial del Proteome Profiler Human Apoptosis Array Kit (ARY009)

Cada proteína ocupa 2 spots duplicados (pares verticales).
Columnas: A (derecha), B, C, D, E (izquierda)
Filas: 1-24 (arriba → abajo)

Reference Spots (según datasheet oficial R&D Systems ARY009):
  - A1, A2  → esquina superior derecha
  - A23, A24 → esquina inferior derecha
  - E1, E2  → esquina superior izquierda
"""

# Mapeo de coordenada (columna, fila) → nombre de proteína/control
MAPA_PROTEINAS = {
    # ── Reference Spots (3 esquinas según datasheet oficial ARY009) ──
    ("A", 1): "Reference", ("A", 2): "Reference",   # esquina superior derecha
    ("A", 23): "Reference", ("A", 24): "Reference", # esquina inferior derecha
    ("E", 1): "Reference", ("E", 2): "Reference",   # esquina superior izquierda

    # ── PBS — Control Negativo ──
    ("D", 23): "PBS", ("D", 24): "PBS",

    # ── Columna B (filas 1-24) ──
    ("B", 1): "Bad", ("B", 2): "Bad",
    ("B", 3): "Bax", ("B", 4): "Bax",
    ("B", 5): "Bcl-2", ("B", 6): "Bcl-2",
    ("B", 7): "Bcl-x", ("B", 8): "Bcl-x",
    ("B", 9): "Pro-Caspase-3", ("B", 10): "Pro-Caspase-3",
    ("B", 11): "Cleaved Caspase-3", ("B", 12): "Cleaved Caspase-3",
    ("B", 13): "Catalase", ("B", 14): "Catalase",
    ("B", 15): "cIAP-1", ("B", 16): "cIAP-1",
    ("B", 17): "cIAP-2", ("B", 18): "cIAP-2",
    ("B", 19): "Claspin", ("B", 20): "Claspin",
    ("B", 21): "Clusterin", ("B", 22): "Clusterin",
    ("B", 23): "Cytochrome c", ("B", 24): "Cytochrome c",

    # ── Columna C (filas 1-24) ──
    ("C", 1): "TRAIL R1/DR4", ("C", 2): "TRAIL R1/DR4",
    ("C", 3): "TRAIL R2/DR5", ("C", 4): "TRAIL R2/DR5",
    ("C", 5): "FADD", ("C", 6): "FADD",
    ("C", 7): "Fas/TNFRSF6/CD95", ("C", 8): "Fas/TNFRSF6/CD95",
    ("C", 9): "HIF-1a", ("C", 10): "HIF-1a",
    ("C", 11): "HO-1/HMOX1/HSP32", ("C", 12): "HO-1/HMOX1/HSP32",
    ("C", 13): "HO-2/HMOX2", ("C", 14): "HO-2/HMOX2",
    ("C", 15): "HSP27", ("C", 16): "HSP27",
    ("C", 17): "HSP60", ("C", 18): "HSP60",
    ("C", 19): "HSP70", ("C", 20): "HSP70",
    ("C", 21): "HTRA2/Omi", ("C", 22): "HTRA2/Omi",
    ("C", 23): "Livin", ("C", 24): "Livin",

    # ── Columna D (filas 1-22) ──
    ("D", 1): "PON2", ("D", 2): "PON2",
    ("D", 3): "p21/CIP1/CDKN1A", ("D", 4): "p21/CIP1/CDKN1A",
    ("D", 5): "p27/Kip1", ("D", 6): "p27/Kip1",
    ("D", 7): "Phospho-p53 (S15)", ("D", 8): "Phospho-p53 (S15)",
    ("D", 9): "Phospho-p53 (S46)", ("D", 10): "Phospho-p53 (S46)",
    ("D", 11): "Phospho-p53 (S392)", ("D", 12): "Phospho-p53 (S392)",
    ("D", 13): "Phospho-Rad17 (S635)", ("D", 14): "Phospho-Rad17 (S635)",
    ("D", 15): "SMAC/Diablo", ("D", 16): "SMAC/Diablo",
    ("D", 17): "Survivin", ("D", 18): "Survivin",
    ("D", 19): "TNF RI/TNFRSF1A", ("D", 20): "TNF RI/TNFRSF1A",
    ("D", 21): "XIAP", ("D", 22): "XIAP",
}

# Lista de proteínas únicas (sin Reference ni PBS)
PROTEINAS_UNICAS = sorted(set(
    v for v in MAPA_PROTEINAS.values() if v not in ("Reference", "PBS")
))

# Clasificación biológica
PRO_APOPTOTICAS = [
    "Bad", "Bax", "Cytochrome c", "Pro-Caspase-3", "Cleaved Caspase-3",
    "SMAC/Diablo", "FADD", "TRAIL R1/DR4", "TRAIL R2/DR5",
    "Fas/TNFRSF6/CD95", "TNF RI/TNFRSF1A",
    "Phospho-p53 (S15)", "Phospho-p53 (S46)", "Phospho-p53 (S392)",
    "Phospho-Rad17 (S635)", "HTRA2/Omi", "HIF-1a",
]

ANTI_APOPTOTICAS = [
    "Bcl-2", "Bcl-x", "Survivin", "XIAP", "cIAP-1", "cIAP-2",
    "Livin", "Claspin", "Clusterin", "Catalase",
    "HO-1/HMOX1/HSP32", "HO-2/HMOX2",
    "HSP27", "HSP60", "HSP70",
    "p21/CIP1/CDKN1A", "p27/Kip1", "PON2",
]

# Pares de duplicados: proteína → lista de coordenadas (col, fila)
PARES_DUPLICADOS = {}
for coord, nombre in MAPA_PROTEINAS.items():
    PARES_DUPLICADOS.setdefault(nombre, []).append(coord)

# Coordenadas de Reference Spots
COORDS_REFERENCIA = [c for c, n in MAPA_PROTEINAS.items() if n == "Reference"]
COORDS_PBS = [c for c, n in MAPA_PROTEINAS.items() if n == "PBS"]

# Número total de spots esperados por strip
# Cols B,C,D tienen 24 filas cada una = 72 spots
# Col A tiene 4 spots (refs: A1,A2,A23,A24)
# Col E tiene 2 spots (refs: E1,E2)
# PBS: 2 spots en D23, D24
TOTAL_SPOTS_ESPERADOS = 78
