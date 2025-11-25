# import json
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

#ARCHIVO_JSON = "escenarios_ahorro.json"

# ------------------ Función de guardado ------------------
'''
def guardar_json(datos):
    """Guarda los datos del escenario en un archivo JSON."""
    try:
        with open(ARCHIVO_JSON, "r") as f:
            escenarios = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        escenarios = []

    escenarios.append(datos)

    with open(ARCHIVO_JSON, "w") as f:
        json.dump(escenarios, f, indent=4)
'''

# ------------------ Cálculo (incluye vf_given_i, f y bisección) ------------------
def vf_given_i(i_p, V0, A, n):
    """Calcula S_n usando aportes al inicio y luego interés cada periodo."""
    S = 0.0
    for t in range(1, n + 1):
        aporte = V0 if t == 1 else A
        S = (S + aporte) * (1 + i_p)
    return S

def f(i_p, V0, A, n, Vf):
    """Función objetivo para encontrar i_p."""
    return vf_given_i(i_p, V0, A, n) - Vf

def find_rate_bisection(V0, A, n, Vf, a=0.0, b=1.0, tol=1e-12, maxiter=200):
    """Método de la bisección para encontrar la tasa periódica para los parámetros dados."""
    try:
        fa = f(a, V0, A, n, Vf)
        fb = f(b, V0, A, n, Vf)
    except Exception:
        return None

    # Intento de expandir límite superior si no hay cambio de signo
    expand_count = 0
    while fa * fb > 0 and expand_count < 50:
        b *= 2
        fb = f(b, V0, A, n, Vf)
        expand_count += 1

    if fa * fb > 0:
        return None  # quien llame mostrará el error

    # Bisección
    m = None
    for _ in range(maxiter):
        m = (a + b) / 2
        fm = f(m, V0, A, n, Vf)
        if abs(fm) < tol or (b - a) / 2 < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return m  # último valor

def parse_float(s):
    """Convierte una cadena a float aceptando coma o punto como separador decimal."""
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip()
    if not s:
        raise ValueError("Cadena vacía")
    s = s.replace(" ", "")  # eliminar espacios
    s = s.replace(",", ".")
    return float(s)


# ------------------ Lógica principal vinculada al GUI ------------------
def calcular_tasa_y_tabla():
    """Calcula la tasa de interés anual y genera la tabla de evolución."""
    try:
        nombre = entry_nombre.get().strip()
        V0 = parse_float(entry_deposito.get())
        Vf = parse_float(entry_final.get())
        A = parse_float(entry_aporte.get())
        n = int(entry_duracion.get())
        periodo = combo_periodo.get().lower()

        if n <= 0:
            messagebox.showerror("Error", "La duración debe ser un número entero positivo.")
            return
        
        if V0 < 50:
            messagebox.showerror("Error", "El depósito inicial debe ser al menos $50.")
            return
        
        if A < 5:
            messagebox.showerror("Error", "El aporte por período debe ser al menos $5.")
            return
        
        if Vf <= V0:
            messagebox.showerror("Error", "El valor final deseado debe ser mayor que el depósito inicial.")
            return

        if not nombre:
            messagebox.showerror("Error", "Por favor ingresa un nombre de usuario.")
            return
        
        if not re.match(r'^[A-Za-zÁÉÍÓÚáéíóúÑñ ]+$', nombre):
            messagebox.showerror("Error", "El nombre solo puede contener letras y espacios.")
            return

        # Periodos anuales según selección
        if periodo == "semanal":
            periodos_anuales = 52
        elif periodo == "mensual":
            periodos_anuales = 12
        elif periodo == "bimestral":
            periodos_anuales = 6
        elif periodo == "trimestral":
            periodos_anuales = 4
        else:
            messagebox.showerror("Error", "Selecciona un periodo válido.")
            return

        # Buscar tasa periódica i_p con bisección
        i_p_mid = find_rate_bisection(V0, A, n, Vf, a=0.0, b=0.5, tol=1e-12, maxiter=300)
        if i_p_mid is None:
            # Intentar con intervalo más grande y si no, error
            i_p_mid = find_rate_bisection(V0, A, n, Vf, a=-0.9, b=5.0, tol=1e-12, maxiter=400)
            if i_p_mid is None:
                messagebox.showerror("Error", "No se encontró una tasa que produzca el valor final dado. Revisa los datos.")
                return

        # Tasa anual efectiva
        tasa_anual_efectiva = (1 + i_p_mid) ** periodos_anuales - 1
        tasa_anual_percent = tasa_anual_efectiva * 100

        label_resultado.config(
            text=f"Tasa de interés anual estimada: {tasa_anual_percent:.4f}%  (tasa periódica i = {i_p_mid:.6f})"
        )

        # Crear tabla de evolución
        saldo = 0.0
        datos_tabla = []

        for t in range(1, n + 1):
            aporte = V0 if t == 1 else A
            capital = saldo + aporte
            interes = capital * i_p_mid
            saldo_final = capital + interes

            datos_tabla.append((t, capital, aporte, interes, saldo_final))
            saldo = saldo_final

        # Mostrar tabla en interfaz (con zebra stripes)
        for i in tabla.get_children():
            tabla.delete(i)

        for idx, fila in enumerate(datos_tabla):
            tag = "oddrow" if idx % 2 == 0 else "evenrow"
            tabla.insert("", "end",
                         values=[fila[0],
                                 f"{fila[1]:.2f}",
                                 f"{fila[2]:.2f}",
                                 f"{fila[3]:.2f}",
                                 f"{fila[4]:.2f}"],
                         tags=(tag,))

        # Graficar evolución
        graficar_evolucion(datos_tabla, tasa_anual_percent)

        # Guardar en JSON
        #guardar_json({
         #   "usuario": nombre,
          #  "deposito_inicial": V0,
           # "valor_final_deseado": Vf,
            #"aporte": A,
            #"periodo": periodo,
            #"duracion": n,
            #"tasa_interes_anual_%": round(tasa_anual_percent, 6),
            #"tasa_periodica_i": round(i_p_mid, 9)
        #})

    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos.")


# ------------------ Gráfica mejorada ------------------
# Estado global sencillo para control de etiquetas
root = tk.Tk()
root.title("Calculadora de Tasa de Interés Anual")
root.geometry("1000x750")
root.resizable(True, True)

mostrar_etiquetas = tk.BooleanVar(value=False)

def graficar_evolucion(datos_tabla, tasa):
    """Genera las gráficas en la interfaz (versión compacta)"""
    periodos = [fila[0] for fila in datos_tabla]
    saldos = [fila[4] for fila in datos_tabla]

    fig, ax = plt.subplots(figsize=(6, 3.5), tight_layout=True)
    ax.plot(periodos, saldos, marker="o", linewidth=1.6)
    ax.set_title(f"Evolución del saldo (Tasa anual {tasa:.2f}%)", fontsize=10)
    ax.set_xlabel("Período")
    ax.set_ylabel("Saldo ($)")
    ax.grid(True, alpha=0.25)

    # Línea horizontal en el valor final
    ax.axhline(saldos[-1], linestyle="--", alpha=0.5, label=f"Final = {saldos[-1]:.2f}")
    ax.legend(fontsize=8)

    # Etiquetas en los puntos (opcional)
    if mostrar_etiquetas.get():
        for x, y in zip(periodos, saldos):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,2), ha="center", fontsize=6)

    # Limpiar frame
    for widget in frame_grafica.winfo_children():
        widget.destroy()

    # Canvas principal (compacto)
    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Botones debajo del gráfico: ver grande y toggle etiquetas
    controls_frame = ttk.Frame(frame_grafica)
    controls_frame.pack(fill="x", pady=(6, 2))

    btn_ver = ttk.Button(controls_frame, text="Ver gráfica completa",
                         command=lambda: mostrar_grafica_grande(periodos, saldos, tasa))
    btn_ver.pack(side="left", padx=(2, 6))

    chk = ttk.Checkbutton(controls_frame, text="Mostrar etiquetas", variable=mostrar_etiquetas,
                          command=lambda: graficar_evolucion(datos_tabla, tasa))
    chk.pack(side="left")


def mostrar_grafica_grande(periodos, saldos, tasa):
    """Muestra la gráfica en una ventana independiente más grande."""
    win = tk.Toplevel(root)
    win.title("Gráfica Ampliada")
    win.geometry("950x700")

    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
    ax.plot(periodos, saldos, marker="o", linewidth=2)
    ax.set_title(f"Evolución del saldo (Tasa anual {tasa:.2f}%)", fontsize=12)
    ax.set_xlabel("Período")
    ax.set_ylabel("Saldo ($)")
    ax.grid(True, alpha=0.25)
    ax.axhline(saldos[-1], linestyle="--", alpha=0.5, label=f"Final = {saldos[-1]:.2f}")
    ax.legend(fontsize=9)

    # Añadir etiquetas en la gráfica grande si la opción está activa
    if mostrar_etiquetas.get():
        for x, y in zip(periodos, saldos):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,2), ha="center", fontsize=9)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ------------------ Interfaz (claro moderno) ------------------


# Paleta clara moderna
BG = "#F6F9FF"       # fondo principal
CARD = "#FFFFFF"     # tarjetas / frames
ACCENT = "#2B6CB0"   # azul acento
ROW_A = "#EAF2FF"    # zebra 1
ROW_B = "#FFFFFF"    # zebra 2

root.configure(bg=BG)
style = ttk.Style()
style.theme_use("clam")

# Estilos generales
style.configure("TFrame", background=BG)
style.configure("Card.TFrame", background=CARD, relief="flat")
style.configure("TLabel", background=BG, font=("Segoe UI", 10))
style.configure("Header.TLabel", background=BG, font=("Segoe UI", 11, "bold"))
style.configure("Accent.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
style.configure("TButton", padding=6)
style.configure("TEntry", padding=4)
style.configure("TCombobox", padding=4)

# Frames principales
outer = ttk.Frame(root, style="TFrame", padding=10)
outer.pack(fill="both", expand=True)

frame_izq = ttk.Frame(outer, style="Card.TFrame", padding=12)
frame_izq.pack(side="left", fill="y", padx=(0, 10), pady=6)

frame_der = ttk.Frame(outer, style="Card.TFrame", padding=12)
frame_der.pack(side="right", fill="both", expand=True, pady=6)

# Campos de entrada (izquierda)
ttk.Label(frame_izq, text="Calculadora de Ahorro", style="Header.TLabel").grid(row=0, column=0, columnspan=2, pady=(0,8))

ttk.Label(frame_izq, text="Nombre del usuario:").grid(row=1, column=0, sticky="w", pady=3)
entry_nombre = ttk.Entry(frame_izq, width=18)
entry_nombre.grid(row=1, column=1, pady=3)

ttk.Label(frame_izq, text="Depósito inicial ($):").grid(row=2, column=0, sticky="w", pady=3)
entry_deposito = ttk.Entry(frame_izq, width=18)
entry_deposito.grid(row=2, column=1, pady=3)

ttk.Label(frame_izq, text="Valor final deseado ($):").grid(row=3, column=0, sticky="w", pady=3)
entry_final = ttk.Entry(frame_izq, width=18)
entry_final.grid(row=3, column=1, pady=3)

ttk.Label(frame_izq, text="Aporte por período ($):").grid(row=4, column=0, sticky="w", pady=3)
entry_aporte = ttk.Entry(frame_izq, width=18)
entry_aporte.grid(row=4, column=1, pady=3)

ttk.Label(frame_izq, text="Duración (n períodos):").grid(row=5, column=0, sticky="w", pady=3)
entry_duracion = ttk.Entry(frame_izq, width=18)
entry_duracion.grid(row=5, column=1, pady=3)

ttk.Label(frame_izq, text="Periodo de aporte:").grid(row=6, column=0, sticky="w", pady=3)
combo_periodo = ttk.Combobox(frame_izq, values=["Semanal", "Mensual", "Bimestral", "Trimestral"], width=16)
combo_periodo.grid(row=6, column=1, pady=3)
combo_periodo.current(0)

ttk.Button(frame_izq, text="Calcular tasa y tabla", command=calcular_tasa_y_tabla).grid(
    row=7, column=0, columnspan=2, pady=(10,6), ipadx=10
)

label_resultado = ttk.Label(frame_izq, text="", style="Accent.TLabel")
label_resultado.grid(row=8, column=0, columnspan=2, pady=(6,2))

# Tabla de resultados (derecha, arriba)
cols = ("Periodo", "Saldo inicial", "Aporte", "Interés", "Saldo final")
tabla = ttk.Treeview(frame_der, columns=cols, show="headings", height=10)
for c in cols:
    tabla.heading(c, text=c)
    tabla.column(c, width=110, anchor="center")
tabla.pack(pady=10, fill="x")

# Estilos zebra para la tabla
style.configure("Treeview",
                background=ROW_A,
                fieldbackground=ROW_B,
                rowheight=24,
                font=("Segoe UI", 10))
style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

tabla.tag_configure("oddrow", background=ROW_A)
tabla.tag_configure("evenrow", background=ROW_B)

# Frame para la gráfica (derecha, abajo)
frame_grafica = ttk.Frame(frame_der, style="Card.TFrame", padding=6)
frame_grafica.pack(fill="both", expand=True)
# Pie de página
pie = ttk.Label(root, text="-EPN-", style="TLabel")
pie.pack(side="bottom", pady=(2,6))

root.mainloop()

