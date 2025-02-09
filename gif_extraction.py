from PIL import Image

# Cargar el GIF
gif_path = "images.gif"  
img = Image.open(gif_path)

# Extraer y guardar cada fotograma
frame_number = 0
while True:
    frame_path = f"frame_{frame_number}.png"
    img.save(frame_path, format="PNG")  
    print(f"Guardado: {frame_path}")
    
    frame_number += 1
    try:
        img.seek(img.tell() + 1)  # Avanzar al siguiente fotograma
    except EOFError:
        break  # Salir cuando no haya más fotogramas
    
    

