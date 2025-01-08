import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imagenes = [
    "pajarito1", "pajarito2", "cocodrilo", "arania", 
    "pastor", "labrador_blanco", "caniche", "armino", 
    "comedor", "carro_basura"
]

# Crear la figura y subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 filas y 5 columnas

# Ajustar la figura para ocupar toda la pantalla
fig.tight_layout()

# Leer y mostrar cada imagen
for i, img_name in enumerate(imagenes):
    try:
        imagen = mpimg.imread(f"../Imagenes_ImageNet/{img_name}.JPEG")
        row, col = divmod(i, 5)  # Calcular fila y columna para cada subplot
        axes[row, col].imshow(imagen)
        axes[row, col].axis("off")  # Quitar los ejes
        axes[row, col].set_title(img_name, fontsize=12)  # Título con el nombre de la imagen
    except FileNotFoundError:
        print(f"No se encontró la imagen: {img_name}.JPEG")

# Mostrar las imágenes
plt.show()
