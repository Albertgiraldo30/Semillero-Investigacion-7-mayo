import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mapa_array import MAPA_PROTEINAS, PRO_APOPTOTICAS, ANTI_APOPTOTICAS

fig, ax = plt.subplots(figsize=(10, 15))

filas = 24
columnas = ['A', 'B', 'C', 'D', 'E']

ax.set_xlim(-0.5, 4.5)
ax.set_ylim(24.5, 0.5) # Y invertido para que la fila 1 quede arriba

# Dibujar la grilla
for col_idx, col in enumerate(columnas):
    for fila in range(1, filas + 1):
        x = col_idx
        y = fila
        
        # Buscar qué hay en esta coordenada
        key = (col, fila)
        proteina = MAPA_PROTEINAS.get(key, "")
        
        # Determinar color
        color_fondo = "#ffffff"
        color_borde = "#dddddd"
        if proteina:
            if proteina == "Reference":
                color_fondo = "#fffde7"
                color_borde = "#fbc02d"
            elif proteina == "PBS":
                color_fondo = "#e0f7fa"
                color_borde = "#00bcd4"
            elif proteina in PRO_APOPTOTICAS:
                color_fondo = "#ffebee"
                color_borde = "#f44336"
            elif proteina in ANTI_APOPTOTICAS:
                color_fondo = "#e3f2fd"
                color_borde = "#2196f3"
            else:
                color_fondo = "#f5f5f5"
                color_borde = "#9e9e9e"
        
        rect = patches.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, 
                                 linewidth=1.5, edgecolor=color_borde, facecolor=color_fondo)
        ax.add_patch(rect)
        
        # Texto
        if proteina:
            nombre = proteina.replace(" ", "\n")
            # Letra más pequeña si el nombre es largo
            fontsize = 7 if len(proteina) > 12 else 8
            if proteina == "Reference": nombre = "Ref"
            ax.text(x, y, nombre, ha='center', va='center', fontsize=fontsize, weight='bold')

# Configurar ejes
ax.set_xticks(range(5))
ax.set_xticklabels(columnas, fontsize=14, weight='bold')
ax.set_yticks(range(1, 25))
ax.set_yticklabels(range(1, 25), fontsize=10)
ax.xaxis.tick_top()

plt.title("Mapa del Kit Proteome Profiler ARY009", fontsize=16, weight='bold', pad=30)
plt.tight_layout()

output_path = "mapa_referencia_ary009.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Mapa generado con éxito en: {output_path}")
