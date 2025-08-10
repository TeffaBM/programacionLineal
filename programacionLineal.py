import re
import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tkinter import Tk, filedialog

# Configuración de Tesseract (ajusta si es necesario)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Función para seleccionar imagen
def seleccionar_imagen():
    root = Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona la imagen del problema",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")]
    )
    return ruta

# Leer imagen y extraer texto
def extraer_texto(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texto = pytesseract.image_to_string(gray, lang="spa+eng")
    return texto.strip()

# Analizar texto
def analizar_texto(texto):
    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    
    # Función objetivo
    fo_linea = next((l for l in lineas if re.search(r"max|min|Z\s*=", l, re.I)), None)
    fo = []
    if fo_linea:
        terminos = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*[a-zA-Z]", fo_linea)
        fo = [float(t) for t in terminos]
    
    # Restricciones
    restricciones = []
    rhs_vals = []
    for l in lineas:
        if re.search(r"<=|≥|>=|≤|=", l) and not re.search(r"max|min|Z", l, re.I):
            coefs = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*[a-zA-Z]", l)
            coefs = [float(c) for c in coefs]
            rhs = re.findall(r"(?:<=|≥|>=|≤|=)\s*([+-]?\d+(?:\.\d+)?)", l)
            if rhs:
                rhs_vals.append(float(rhs[0]))
            restricciones.append(coefs)
    
    return fo, restricciones, rhs_vals

# Resolver Simplex
def resolver_simplex(fo, restricciones, rhs):
    c = [-x for x in fo]  # scipy minimiza, cambiamos signo
    res = linprog(c, A_ub=restricciones, b_ub=rhs, method='highs')
    return res

# Graficar
def graficar_solucion(restricciones, rhs, resultado):
    x = np.linspace(0, max(rhs)*1.2, 200)
    plt.figure(figsize=(6,6))

    for (a,b), c in zip(restricciones, rhs):
        y = (c - a*x) / b
        y[y < 0] = np.nan
        plt.plot(x, y, label=f"{a}x + {b}y <= {c}")
    
    # Región factible
    y_max = np.minimum.reduce([(c - a*x)/b for (a,b), c in zip(restricciones, rhs)])
    y_max[y_max < 0] = np.nan
    plt.fill_between(x, 0, y_max, alpha=0.2, color="green")

    # Punto óptimo
    if resultado.success:
        plt.plot(resultado.x[0], resultado.x[1], 'ro', label=f"Óptimo: {resultado.x}")
    
    plt.xlim(0, max(rhs)*1.2)
    plt.ylim(0, max(rhs)*1.2)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------
# Ejecución
# -------------------------
ruta_img = seleccionar_imagen()
if ruta_img:
    print(f"📂 Imagen seleccionada: {ruta_img}")
    texto = extraer_texto(ruta_img)
    print("\n📄 Texto detectado:\n", texto)

    fo, restricciones, rhs = analizar_texto(texto)
    print("\n📊 Función objetivo:", fo)
    print("📊 Restricciones:", restricciones)
    print("📊 Lados derechos:", rhs)

    resultado = resolver_simplex(fo, restricciones, rhs)
    print("\n✅ Resultado simplex:", resultado)

    if len(fo) == 2:
        graficar_solucion(restricciones, rhs, resultado)
else:
    print("❌ No seleccionaste ninguna imagen.")
