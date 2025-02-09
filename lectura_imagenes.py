import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from tabulate import tabulate

# Ruta de la carpeta con las imágenes
ruta_carpeta = "C:/Users/DANIE/OneDrive/Escritorio/Maestria/Materias/Materias 1er Quatrimestre/Vision Artificial/Act3_Vision/Imagenes_Analisis"
años = list(range(2000, 2020))

# Parámetro de escala (20 km ≈ 51 píxeles)
PIXEL_TO_KM2 = (20 / 51) ** 2

target_color = np.array([133, 120, 49])  # Color objetivo
tolerance = 30  # Rango de tolerancia para detectar el color


import numpy as np
import cv2
from PIL import Image

import cv2
import numpy as np
from PIL import Image

def get_colored_pixels(img, target_color=(133, 120, 49), threshold=50, min_lightness=10, max_lightness=140):
    
    # Convertir a OpenCV y luego a LAB
    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2LAB)

    # Convertir el color objetivo a LAB
    target_lab = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_RGB2LAB)[0][0]

    # Crear los umbrales de color
    lower_bound = np.clip(target_lab - threshold, 0, 255)
    upper_bound = np.clip(target_lab + threshold, 0, 255)

    # Aplicar umbral adicional para evitar blancos
    lower_bound[0] = max(lower_bound[0], min_lightness)  # L (luminosidad) mínimo
    upper_bound[0] = min(upper_bound[0], max_lightness)  # L (luminosidad) máximo

    # Crear máscara
    mask = cv2.inRange(img_cv, lower_bound, upper_bound)

    return mask.astype(np.uint8) * 255  # Convertir a imagen binaria (0 o 255)


def preprocess_image(img):
    """Filtra los píxeles cercanos al color objetivo, convierte a escala de grises y ajusta contraste con CLAHE."""

    if img.mode == "P":
        img = img.convert("RGB")  # Convertir a RGB si es necesario

    img_np = np.array(img)  # Convertir imagen de PIL a NumPy

    # Obtener máscara de los píxeles cercanos al color objetivo
    mask = get_colored_pixels(img)

    # Aplicar la máscara para conservar solo los píxeles filtrados
    img_filtered = cv2.bitwise_and(img_np, img_np, mask=mask)

    # Convertir a escala de grises solo la parte filtrada
    img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2GRAY)

    # Aplicar CLAHE para mejorar contraste sin sobreexponer
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img_gray)

    return img_eq



def segment_image(img):
    """Segmenta la imagen usando Otsu Adaptativo."""
    img_blur = cv2.GaussianBlur(img, (3,3), 0)  # Reduce ruido sin perder bordes

    # 1. Aplicar umbral adaptativo (simula Otsu por regiones)
    adaptive_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 2  # Tamaño de bloque más grande y una constante ajustada
    )

    # 2. Aplicar Otsu sobre la imagen suavizada para obtener una referencia global
    _, otsu_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Combinación de ambos métodos (Otsu sirve como máscara para el adaptativo)
    combined_thresh = cv2.bitwise_and(adaptive_thresh, otsu_thresh)

    # 4. Eliminación de ruido con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    otsu_clean = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)

    return otsu_clean,combined_thresh

def calculate_deforested_area(binary_img):
    """Calcula el área deforestada en km²."""
    deforested_pixels = np.sum(binary_img == 255)
    area_km2 = deforested_pixels * PIXEL_TO_KM2
    return area_km2

imagenes = []
results = []
for archivo in os.listdir(ruta_carpeta):
    if archivo.endswith(".png"):
        img = Image.open(os.path.join(ruta_carpeta, archivo))
        imagenes.append((archivo, img))

# Función para extraer el número del nombre del archivo
def extraer_numero(archivo):
    match = re.search(r"(\d+)", archivo)
    return int(match.group(1)) if match else 0

# Ordenar las imágenes por número extraído
imagenes.sort(key=lambda x: extraer_numero(x[0]))

# Procesar cada imagen y mostrar en subplot
for i,(nombre, imagen) in enumerate(imagenes):
    #print(f"\nProcesando: {nombre}")
    if i >= len(años):  # Evita errores si hay más imágenes que años
        break
    año = años[i]
    img_eq = preprocess_image(imagen)
    binary_img, otsu_sucio  = segment_image(img_eq)
    area = calculate_deforested_area(binary_img)
    
    # Guardar nombre y área en la lista de resultados
    results.append((nombre,año, area))
    
    # Mostrar en un subplot
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    
    axs[0].imshow(imagen)
    axs[0].set_title("Original")
    axs[0].axis("off")
    
    axs[1].imshow(img_eq, cmap="gray")
    axs[1].set_title("Preprocesada (CLAHE)")
    axs[1].axis("off")
    
    axs[2].imshow(binary_img, cmap="gray")
    axs[2].set_title(f"Segmentada (Área: {area:.2f} km²)")
    axs[2].axis("off")
    
    axs[3].imshow(otsu_sucio, cmap="gray")
    axs[3].set_title(f"Otsu sin Limpiar")
    axs[3].axis("off")
    
    plt.suptitle(f"Análisis de imagen: {nombre}")
    plt.show()


# Mostrar resultados
for nombre,años, area in results:
    print(f"{nombre}: Área deforestada = {area:.2f} km²")
    
final_data = pd.DataFrame(results, columns=["Nombre de Imagen", "Año", "Área Deforestada (km²)"])
print(tabulate(final_data, headers="keys", tablefmt="fancy_grid"))


# Guardar los resultados en un archivo Excel
ruta_salida = os.path.join(ruta_carpeta, "resultados_deforestacion.xlsx")
final_data.to_excel(ruta_salida, index=False)

print(f"Resultados guardados en: {ruta_salida}")




# Generar histograma
plt.figure(figsize=(12, 6))
plt.bar(final_data["Año"], final_data["Área Deforestada (km²)"], color="#FF8C00", edgecolor="black", label="Área Deforestada")

# Agregar línea de tendencia
x = final_data["Año"]
y = final_data["Área Deforestada (km²)"]
z = np.polyfit(x, y, 1)  # Ajuste de línea recta (grado 1)
p = np.poly1d(z)  # Función polinómica

plt.plot(x, p(x), linestyle="--", color="blue", linewidth=2, label="Tendencia")  # Línea de tendencia en rojo

# Configuración del gráfico
plt.xlabel("Año")
plt.ylabel("Área Deforestada (km²)")
plt.title("Evolución de la Deforestación por Año")
plt.xticks(np.arange(min(x), max(x) + 1, 1), rotation=90)  # Años en vertical
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.show()




# Supongamos que final_data ya está definido con las columnas "Año" y "Área Deforestada (km²)"
x = final_data["Año"]
y = final_data["Área Deforestada (km²)"]

# Ajuste de regresión lineal
z = np.polyfit(x, y, 1)  # Grado 1 (recta)
p = np.poly1d(z)  # Función polinómica

# Cálculo de métricas
pendiente = z[0]  # Tasa de crecimiento anual promedio
intercepto = z[1]

# R² (coeficiente de determinación)
y_pred = p(x)
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

# Desviación estándar
std_dev = np.std(y)

# Año con mayor y menor deforestación
año_max = final_data.loc[y.idxmax(), "Año"]
año_min = final_data.loc[y.idxmin(), "Año"]

# Resultados
metricas = {
    "Pendiente (Tasa de Crecimiento Anual Promedio)": pendiente,
    "Intercepto": intercepto,
    "Coeficiente de Determinación (R²)": r2,
    "Desviación Estándar": std_dev,
    "Año con Mayor Deforestación": año_max,
    "Año con Menor Deforestación": año_min
}

for k, v in metricas.items():
    print(f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}")
