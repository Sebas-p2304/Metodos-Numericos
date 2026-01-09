# %%

def solucionar_sistema(*ecuaciones):
    import numpy as np
    import sympy as sp

    # Convertir ecuaciones a expresiones simbólicas
    expresiones = []
    for eq in ecuaciones:
        izquierda, derecha = eq.split('=')
        expresiones.append(sp.sympify(izquierda) - sp.sympify(derecha))

    # Extraer variables
    variables = sorted(
        list(set().union(*[expr.free_symbols for expr in expresiones])),
        key=lambda x: x.name
    )

    n = len(variables)
    m = len(expresiones)

    # Verificar solución única
    if n != m:
        return expresiones, None, "El sistema no tiene solución única."

    A = []
    b = []

    for expr in expresiones:
        fila = [expr.coeff(var) for var in variables]
        A.append(fila)
        b.append(-expr.subs({var: 0 for var in variables}))

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    det = np.linalg.det(A)
    if np.isclose(det, 0):
        return expresiones, None, "El sistema no tiene solución única."

    sol = np.linalg.solve(A, b)

    solucion = {str(var): sol[i] for i, var in enumerate(variables)}

    return expresiones, solucion, None

# %%
def graficar_sistema(expresiones, solucion, x_range=(-10, 10), puntos=400):
    import numpy as np
    import matplotlib.pyplot as plt

    variables = sorted(
        list(set().union(*[expr.free_symbols for expr in expresiones])),
        key=lambda x: x.name
    )

    if len(variables) != 2:
        print("La función solo puede graficar sistemas con dos variables.")
        return

    x_var, y_var = variables
    x_vals = np.linspace(x_range[0], x_range[1], puntos)

    plt.figure(figsize=(8, 6))

    for expr in expresiones:
        a = expr.coeff(x_var)
        b = expr.coeff(y_var)
        c = expr.subs({x_var: 0, y_var: 0})

        if b == 0:
            x_const = -c / a
            plt.axvline(float(x_const), linestyle='--')
        else:
            y_vals = (-a * x_vals - c) / b
            plt.plot(x_vals, y_vals)

    if solucion is not None:
        x_sol = solucion[str(x_var)]
        y_sol = solucion[str(y_var)]
        plt.scatter(x_sol, y_sol, color='red', zorder=5)
        plt.text(x_sol, y_sol, f' ({x_sol:.2f}, {y_sol:.2f})')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(alpha=0.4)

    plt.xlabel(str(x_var))
    plt.ylabel(str(y_var))
    plt.title('Sistema de ecuaciones lineales')

    plt.show()



# %%
def resolver_y_graficar(*ecuaciones):
    expresiones, solucion, error = solucionar_sistema(*ecuaciones)

    if error is not None:
        print(error)
        return

    graficar_sistema(expresiones, solucion)


# %%
import numpy as np
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())


def eliminacion_gaussiana_redondeo(A: np.ndarray | list[list[float | int]]) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante eliminación gaussiana
    SIN pivoteo y usando aritmética de redondeo a 2 decimales.
    """

    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)

    assert A.shape[0] == A.shape[1] - 1, \
        "La matriz A debe ser de tamaño n-by-(n+1)."

    n = A.shape[0]

    # ------------------------------------------------
    # Eliminación hacia adelante (sin intercambios)
    # ------------------------------------------------
    for i in range(n - 1):

        if A[i, i] == 0:
            raise ValueError(
                f"Pivote nulo en la posición ({i},{i}). "
                "El método sin pivoteo no puede continuar."
            )

        for j in range(i + 1, n):
            m = round(A[j, i] / A[i, i], 2)

            for k in range(i, n + 1):
                A[j, k] = round(A[j, k] - m * A[i, k], 2)

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    # ------------------------------------------------
    # Sustitución hacia atrás (con redondeo)
    # ------------------------------------------------
    solucion = np.zeros(n)

    solucion[n - 1] = round(A[n - 1, n] / A[n - 1, n - 1], 2)

    for i in range(n - 2, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma = round(suma + A[i, j] * solucion[j], 2)

        solucion[i] = round(
            (A[i, n] - suma) / A[i, i],
            2
        )

    return solucion


# %%
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np
def eliminacion_gaussiana(A: np.ndarray | list[list[float | int]]) -> np.ndarray:
 
    sumas_restas = 0
    mult_div = 0
    intercambios = 0

    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)

    assert A.shape[0] == A.shape[1] - 1, \
        "La matriz A debe ser de tamaño n-by-(n+1)."

    n = A.shape[0]

    # ------------------------------------------------
    # Eliminación hacia adelante
    # ------------------------------------------------
    for i in range(n - 1):
        # Buscar pivote
        p = None
        for pi in range(i, n):
            if A[pi, i] != 0:
                if p is None or abs(A[pi, i]) < abs(A[p, i]):
                    p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            A[[i, p], :] = A[[p, i], :]
            intercambios += 1

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            for k in range(i, n + 1):
                A[j, k] = A[j, k] - m * A[i, k]

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    # ------------------------------------------------
    # Sustitución hacia atrás
    # ------------------------------------------------
    solucion = np.zeros(n)

    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    mult_div += 1

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
        solucion[i] = (A[i, n] - suma) / A[i, i]
 

    print("\nGauss:")
    if intercambios > 0:
        print("Sí se necesitan intercambios de fila")
    print('la solución es:')
    return  solucion
    

# %%
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np
def eliminacion_gaussiana_redondeo32bits(A: np.ndarray | list[list[float | int]]) -> np.ndarray:
 
    sumas_restas = 0
    mult_div = 0
    intercambios = 0

    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=np.float32)

    assert A.shape[0] == A.shape[1] - 1, \
        "La matriz A debe ser de tamaño n-by-(n+1)."

    n = A.shape[0]

    # ------------------------------------------------
    # Eliminación hacia adelante
    # ------------------------------------------------
    for i in range(n - 1):
        # Buscar pivote
        p = None
        for pi in range(i, n):
            if A[pi, i] != 0:
                if p is None or abs(A[pi, i]) < abs(A[p, i]):
                    p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            A[[i, p], :] = A[[p, i], :]
            intercambios += 1

        for j in range(i + 1, n):
            m = np.float32(A[j, i] / A[i, i])
            for k in range(i, n + 1):
                A[j, k] = np.float32(A[j, k] - m * A[i, k])

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    # ------------------------------------------------
    # Sustitución hacia atrás
    # ------------------------------------------------
    solucion = np.zeros(n)

    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    mult_div += 1

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += np.float32(A[i, j] * solucion[j])
        solucion[i] = np.float32((A[i, n] - suma) / A[i, i])
 

    print("\nGauss con aritmética de 32 bits:")
    print('la solución es:')
    return  solucion
    

# %%
def calcular_error(vec_exacto, vec_aproximado):
    """Calcula el error relativo porcentual entre dos vectores."""
    import numpy as np

    if len(vec_exacto) != len(vec_aproximado):
        raise ValueError("Los vectores deben tener la misma longitud.")

    vec_exacto = np.array(vec_exacto, dtype=float)
    vec_aproximado = np.array(vec_aproximado, dtype=float)

    errores = np.abs((vec_exacto - vec_aproximado) / vec_exacto)*100
    for i in range(len(errores)):
        print('\nError calculado para la variable x',(i+1),':' , errores[i], '%')
    return


import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np
def gauss_jordan_redondeo32bits(A: np.ndarray | list[list[float | int]]) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de Gauss-Jordan
    con contadores de operaciones aritméticas.
    """
    sumas_restas = 0
    mult_div = 0

    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=np.float32)

    assert A.shape[0] == A.shape[1] - 1, \
        "La matriz A debe ser de tamaño n-by-(n+1)."

    n = A.shape[0]

    for i in range(n):
        # ------------------------------------------------
        # Encontrar pivote
        # ------------------------------------------------
        p = None
        for pi in range(i, n):
            if A[pi, i] != 0:
                if p is None or abs(A[pi, i]) < abs(A[p, i]):
                    p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            A[[i, p], :] = A[[p, i], :]

        # ------------------------------------------------
        # Normalizar fila pivote
        # ------------------------------------------------
        pivote = A[i, i]
        for k in range(i, n + 1):
            A[i, k] = np.float32(A[i, k] / pivote)


        # ------------------------------------------------
        # Eliminación arriba y abajo
        # ------------------------------------------------
        for j in range(n):
            if j == i:
                continue
            m = np.float32(A[j, i])
            for k in range(i, n + 1):
                A[j, k] = np.float32(A[j, k] - m * A[i, k])

        logging.info(f"\n{A}")

    solucion = A[:, -1]

    print("\nGauss-Jordan:")
    print('la solución es con aritmetica de 32 bits:\n')

    return solucion
# %%



