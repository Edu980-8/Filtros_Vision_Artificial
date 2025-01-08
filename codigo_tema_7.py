import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import convolve

# Definir el kernel de filtro paso alto (realce de bordes)
high_pass_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

def aplicar_filtro_paso_alto(ruta_imagen):
    imagen = mpimg.imread(ruta_imagen)
    
    # Convertir a escala de grises si la imagen es RGB
    if len(imagen.shape) == 3:  # Si tiene 3 canales (color)
        imagen_gris = np.mean(imagen, axis=2)
    else:
        imagen_gris = imagen
    
    # Aplicar el filtro paso alto
    imagen_filtrada = convolve(imagen_gris, high_pass_kernel, mode='constant', cval=0.0)
    
    # Mostrar la imagen original
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen_gris, cmap="gray")
    plt.title("Imagen Original")
    plt.axis("off")
    
    # Mostrar la imagen filtrada
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap="gray")
    plt.title("Filtro Paso Alto")
    plt.axis("off")
    
    plt.show()

# Lista de imágenes disponibles
imagenes = [
    "pajarito1", "pajarito2", "cocodrilo", "arania", "pastor", 
    "labrador_blanco", "caniche", "armino", "comedor", "carro_basura"
]

# Mostrar las opciones de imágenes
print("Imágenes disponibles:")
for i, nombre in enumerate(imagenes, start=1):
    print(f"{i}. {nombre}")

# Solicitar al usuario que elija la imagen
try:
    seleccion = int(input("Elija el número de la imagen que desea ver filtrada: "))
    
    if 1 <= seleccion <= len(imagenes):
        # Obtener el nombre de la imagen seleccionada
        nombre_imagen = imagenes[seleccion - 1]
        ruta = f"../Imagenes_ImageNet/{nombre_imagen}.JPEG"
        
        # Aplicar el filtro a la imagen seleccionada
        aplicar_filtro_paso_alto(ruta)
    else:
        print("Número de imagen no válido.")
except ValueError:
    print("Por favor, ingrese un número válido.")
