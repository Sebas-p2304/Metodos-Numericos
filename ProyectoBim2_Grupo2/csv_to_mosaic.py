import pandas as pd
import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
from stl import mesh
import os


df = pd.read_csv(
    r"C:\Users\sebas\Documents\EPN\2025-B\Metodos\Proyecto_II_Bim\output\elev_all_hgts.csv"
)

lat_min = df["lat"].min()
lat_max = df["lat"].max()
lon_min = df["lon"].min()
lon_max = df["lon"].max()
res = 0.001  # grados (~100 m)
lats = np.arange(lat_max, lat_min, -res)  # de norte a sur
lons = np.arange(lon_min, lon_max, res)

nrows = len(lats)
ncols = len(lons)

print(nrows, ncols)
raster = np.full((nrows, ncols), np.nan)

row_idx = ((lat_max - df["lat"]) / res).astype(int)
col_idx = ((df["lon"] - lon_min) / res).astype(int)

for r, c, h in zip(row_idx, col_idx, df["elev_m"]):
    if 0 <= r < nrows and 0 <= c < ncols:
        raster[r, c] = h


def nanmean_filter(values):
    values = values[~np.isnan(values)]
    return np.mean(values) if len(values) else np.nan

raster_filled = generic_filter(raster, nanmean_filter, size=3)

print("Min elev:", np.nanmin(raster_filled))
print("Max elev:", np.nanmax(raster_filled))

plt.imshow(raster_filled, cmap="terrain")
plt.colorbar(label="Elevación (m)")
plt.title("Mosaico Raster de Elevación")
plt.show()

nrows, ncols = raster_filled.shape

x = np.arange(ncols)
y = np.arange(nrows)
X, Y = np.meshgrid(x, y)
Z = raster_filled
vert_exag = 2.0  # exageración vertical
Z = Z * vert_exag
vertices = []
faces = []

for i in range(nrows):
    for j in range(ncols):
        vertices.append([X[i, j], Y[i, j], Z[i, j]])

vertices = np.array(vertices)

def idx(i, j):
    return i * ncols + j

for i in range(nrows - 1):
    for j in range(ncols - 1):
        v0 = idx(i, j)
        v1 = idx(i, j + 1)
        v2 = idx(i + 1, j)
        v3 = idx(i + 1, j + 1)

        faces.append([v0, v1, v2])
        faces.append([v1, v3, v2])

faces = np.array(faces)
base_height = np.nanmin(Z) - 10  # 10 unidades abajo
terrain = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

for i, face in enumerate(faces):
    for j in range(3):
        terrain.vectors[i][j] = vertices[face[j], :]

#terrain.save("terreno.stl")
#print("STL generado: terreno.stl")
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

stl_path = os.path.join(out_dir, "terreno.stl")
terrain.save(stl_path)

print(f"STL guardado en: {stl_path}")
