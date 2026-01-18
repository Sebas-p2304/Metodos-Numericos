import pandas as pd
import os

# Ruta de la carpeta donde está el CSV
ruta = r'C:\Users\sebas\Documents\EPN\2025-B\Metodos\Proyecto_II_Bim\output'

print(f"¿Existe la carpeta?: {os.path.exists(ruta)}")

# Listar archivos dentro de la carpeta output
archivos = os.listdir(ruta)
print("Archivos en la carpeta output:")
for archivo in archivos:
    print(archivo)

# Ruta completa del archivo CSV
ruta_completa = os.path.join(ruta, 'elev_all_hgts.csv')

print(f"¿Existe el archivo CSV?: {os.path.exists(ruta_completa)}")

# Leer el CSV
df = pd.read_csv(ruta_completa)

# Mostrar las primeras filas
print(df.tail(500))




