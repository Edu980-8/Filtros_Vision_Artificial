import matplotlib.pyplot as plt
import matplotlib.image as mpimg


imagenes = ["pajarito1","pajarito2","cocodrilo","arania","pastor","labrador_blanco","caniche","armino","comedor","carro_basura"]

# Leer la imagen

for i in range(len(imagenes)):
    imagen = mpimg.imread(f"../Imagenes_ImageNet/{imagenes[i]}.JPEG")
    plt.figure(i)
    # Mostrar la imagen
    plt.imshow(imagen)
    plt.axis("off")  # Quitar los ejes

