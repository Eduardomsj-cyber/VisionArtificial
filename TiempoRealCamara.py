import cv2
import numpy as np
import matplotlib.pyplot as plt

# Inicializar el clasificador de rostros
face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_name)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Configurar la ventana de visualización
plt.ion()  # Modo interactivo de matplotlib
fig, ax = plt.subplots()
img_plot = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Actualizar la visualización en tiempo real
        img_plot.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.pause(0.1)  # Pausa para actualizar la ventana

except KeyboardInterrupt:
    pass

# Liberar recursos
cap.release()
plt.ioff()
plt.show()
