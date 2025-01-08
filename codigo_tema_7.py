import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import convolve

# Definir los kernels de los filtros
high_pass_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

low_pass_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16  # Filtro Gaussiano 3x3 normalizado

edge_detector_kernel= np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])

# Filtro de Roberts
roberts_kernel_x = np.array([
    [1, 0],
    [0, -1]
])

roberts_kernel_y = np.array([
    [0, 1],
    [-1, 0]
])

def aplicar_filtro(filtro, ruta_imagen):
    """
    Aplica el filtro seleccionado a la imagen ubicada en la ruta dada.
    
    Parámetros:
    - filtro (ndarray): El filtro a aplicar.
    - ruta_imagen (str): Ruta al archivo de la imagen.
    """
    imagen = mpimg.imread(ruta_imagen)
    
    # Convertir a escala de grises si la imagen es RGB
    if len(imagen.shape) == 3:  # Si tiene 3 canales (color)
        imagen_gris = np.mean(imagen, axis=2)
    else:
        imagen_gris = imagen
    
    # Aplicar el filtro seleccionado
    imagen_filtrada = convolve(imagen_gris, filtro, mode='constant', cval=0.0)
    
    return imagen_filtrada

def seleccionar_imagen(imagenes):
    """Solicitar al usuario que seleccione una imagen."""
    print("Imágenes disponibles:")
    for i, nombre in enumerate(imagenes, start=1):
        print(f"{i}. {nombre}")
    
    try:
        seleccion = int(input("Elija el número de la imagen que desea ver filtrada: "))
        if 1 <= seleccion <= len(imagenes):
            return f"../Imagenes_ImageNet/{imagenes[seleccion - 1]}.JPEG"
        else:
            print("Número de imagen no válido.")
            return None
    except ValueError:
        print("Por favor, ingrese un número válido.")
        return None

def seleccionar_filtro():
    """Solicitar al usuario que seleccione un filtro."""
    print("Filtros disponibles:")
    print("1. Filtro Paso Bajo")
    print("2. Filtro Paso Alto")
    print("3. Filtro de Roberts")
    print("4. Detector de Bordes")
    
    try:
        filtro_seleccionado = int(input("Elija el número del filtro que desea aplicar: "))
        if filtro_seleccionado == 1:
            return low_pass_kernel
        elif filtro_seleccionado == 2:
            return high_pass_kernel
        elif filtro_seleccionado == 3:
            return (roberts_kernel_x, roberts_kernel_y)
        elif filtro_seleccionado == 4:
            return edge_detector_kernel
        else:
            print("Número de filtro no válido.")
            return None
    except ValueError:
        print("Por favor, ingrese un número válido.")
        return None

def mostrar_imagenes(imagen_original, imagen_filtrada):
    """Muestra la imagen original y la filtrada."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen_original, cmap="gray")
    plt.title("Imagen Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap="gray")
    plt.title("Imagen Filtrada")
    plt.axis("off")
    
    plt.show()

def main():
    imagenes = ["pajarito1", "pajarito2", "cocodrilo", "arania", "pastor", 
                "labrador_blanco", "caniche", "armino", "comedor", "carro_basura",
                "perro_desenfoque"]
    
    ruta_imagen = seleccionar_imagen(imagenes)
    if ruta_imagen:
        filtro = seleccionar_filtro()
        if filtro is not None:
            imagen_gris = mpimg.imread(ruta_imagen)
            if isinstance(filtro, tuple):  # Si es un filtro de Roberts
                imagen_filtrada_x = aplicar_filtro(filtro[0], ruta_imagen)
                imagen_filtrada_y = aplicar_filtro(filtro[1], ruta_imagen)
                # Combinar los resultados de las dos direcciones
                imagen_filtrada = np.hypot(imagen_filtrada_x, imagen_filtrada_y)
            else:
                imagen_filtrada = aplicar_filtro(filtro, ruta_imagen)
            mostrar_imagenes(imagen_gris, imagen_filtrada)

if __name__ == "__main__":
    main()
