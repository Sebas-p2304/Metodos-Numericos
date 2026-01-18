#!/usr/bin/env python3
"""
hgt_to_csv_vscode.py

Versión lista para ejecutar desde Visual Studio Code sin argumentos.
Lee TODOS los archivos .hgt desde una carpeta fija (configurada abajo), procesa
los valores válidos y guarda un CSV con columnas: lat, lon, elev_m.

Cómo usar:
- Abre este archivo en VS Code (en la misma máquina donde están los .hgt)
- Presiona F5 o ejecuta la celda/script

Nota: se utilizan rutas con barras / para evitar problemas de escape en cadenas.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import csv
import sys

# ---------------- CONFIGURACIÓN (modifica aquí si hace falta) ----------------
# Usa rutas con / para mayor compatibilidad en la cadena literal
HGT_DIR = Path("C:/Users/sebas/Documents/EPN/2025-B/Metodos/Proyecto_II_Bim/hgt")
SCRIPT_DIR = Path("C:/Users/sebas/Documents/EPN/2025-B/Metodos/Proyecto_II_Bim")
OUT_DIR = SCRIPT_DIR / "output"
OUT_CSV_NAME = "elev_all_hgts.csv"
# ---------------------------------------------------------------------------


def parse_hgt_filename(fname: Path) -> Tuple[float, float]:
    """Extrae lat0 (inferior) y lon0 (izquierda) desde el nombre del fichero.
    Espera nombres tipo: N00W078.hgt, S02W080.hgt, N01E123.hgt
    Devuelve (lat0, lon0) en grados decimales.
    """
    s = fname.stem
    token = s[:7]
    if len(token) < 7:
        raise ValueError(f"Nombre de tile inesperado: {s}")
    lat_hemi = token[0].upper()
    lat_deg = int(token[1:3])
    lon_hemi = token[3].upper()
    lon_deg = int(token[4:7])
    lat0 = lat_deg if lat_hemi == 'N' else -lat_deg
    lon0 = lon_deg if lon_hemi == 'E' else -lon_deg
    return float(lat0), float(lon0)


def read_hgt_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lee un archivo .hgt y retorna lon_grid, lat_grid, elev (2D arrays).

    - Formato: big-endian int16
    - Convierte -32768 a np.nan
    - latitudes de norte a sur
    """
    data = np.fromfile(str(path), dtype='>i2')
    if data.size == 0:
        raise ValueError(f"Archivo vacío: {path}")
    size = int(np.sqrt(data.size))
    if size * size != data.size:
        raise ValueError(f"Tamaño inesperado en {path}; data.size={data.size}")
    elev = data.reshape((size, size)).astype(np.float32)
    elev[elev == -32768] = np.nan

    lat0, lon0 = parse_hgt_filename(path)
    lats = np.linspace(lat0 + 1.0, lat0, size)
    lons = np.linspace(lon0, lon0 + 1.0, size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lon_grid, lat_grid, elev


def filter_hgts_all(hgt_dir: Path):
    """Procesa TODOS los .hgt en hgt_dir y devuelve arrays 1D: lat, lon, elev.
    No se aplica filtrado espacial.
    """
    hgt_dir = Path(hgt_dir)
    if not hgt_dir.exists() or not hgt_dir.is_dir():
        raise FileNotFoundError(f"Directorio no existe: {hgt_dir}")

    hgt_files = sorted(hgt_dir.glob('*.hgt'))
    if not hgt_files:
        raise FileNotFoundError(f"No se encontraron archivos .hgt en {hgt_dir}")

    lon_parts, lat_parts, elev_parts = [], [], []
    total_points = 0

    for hf in hgt_files:
        try:
            lon_grid, lat_grid, elev = read_hgt_file(hf)
        except Exception as e:
            print(f"[WARN] Saltando {hf.name}: {e}")
            continue

        mask = ~np.isnan(elev)
        n = int(mask.sum())
        if n == 0:
            continue
        total_points += n
        lon_parts.append(lon_grid[mask].ravel())
        lat_parts.append(lat_grid[mask].ravel())
        elev_parts.append(elev[mask].ravel())
        print(f"Procesado {hf.name}: puntos válidos = {n}")

    if not lon_parts:
        raise ValueError('No se encontraron datos válidos en los .hgt')

    lon_all = np.concatenate(lon_parts)
    lat_all = np.concatenate(lat_parts)
    elev_all = np.concatenate(elev_parts)

    print(f"Total puntos válidos recogidos: {total_points}")
    return lat_all, lon_all, elev_all


def save_csv_lat_lon_elev(lat, lon, elev, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    idx = np.lexsort((lon, -lat))
    lat_sorted = lat[idx]
    lon_sorted = lon[idx]
    elev_sorted = elev[idx]

    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lat', 'lon', 'elev_m'])
        for la, lo, el in zip(lat_sorted, lon_sorted, elev_sorted):
            writer.writerow([f"{la:.7f}", f"{lo:.7f}", f"{el:.3f}"])

    print(f"CSV guardado en: {out_csv} (filas: {len(lat)})")


def main():
    print('--- HGT -> CSV (todos los tiles) ---')
    print(f'HGT_DIR = {HGT_DIR}')
    print(f'OUT_DIR = {OUT_DIR}')
    try:
        lat_all, lon_all, elev_all = filter_hgts_all(HGT_DIR)
    except Exception as e:
        print('Error al procesar HGTs:', e)
        sys.exit(1)

    out_csv = OUT_DIR / OUT_CSV_NAME
    save_csv_lat_lon_elev(lat_all, lon_all, elev_all, out_csv)
    print('Proceso completado.')


if __name__ == '__main__':
    main()
