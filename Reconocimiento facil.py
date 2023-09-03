import cv2


# Inicializar la cámara
# El cero indica que es la camara frontal
cap = cv2.VideoCapture(0)

# Tomar y guardar la primera foto
ret, user_face = cap.read()
# el nombre de user_face guarda la foto y despues en la parte 23 compara la foto con el mismo nombre
cv2.imwrite('user_face.jpg', user_face)
print("Primera foto tomada y guardada.")

cap.release()

# Esperar y tomar la segunda foto
input("Presiona Enter para tomar la segunda foto...")
cap = cv2.VideoCapture(0)

# Capturar una imagen de la cámara y almacenarla en current_user_face
# alamneca en current_user_face la otra foto con el cual lo comprar
ret, current_user_face = cap.read()

# Comparar las dos fotos
if user_face is not None and current_user_face is not None:
    user_face_gray = cv2.cvtColor(user_face, cv2.COLOR_BGR2GRAY)
    current_user_face_gray = cv2.cvtColor(current_user_face, cv2.COLOR_BGR2GRAY)

#las imágenes en color (user_face y current_user_face) se convierten a escala de grises utilizando la función cv2.cvtColor().
    # La detección de rostros y la comparación de similitud suelen funcionar mejor en imágenes en escala de grises,
    # ya que reducen la complejidad y resaltan mejor las características importantes.
    
    similarity = cv2.matchTemplate(user_face_gray, current_user_face_gray, cv2.TM_CCOEFF_NORMED)
    similarity_threshold = 0.7

    if similarity > similarity_threshold:
        print("Bienvenido, mismo usuario.")
    else:
        print("No eres el mismo usuario.")
else:
    print("No se pudieron capturar las imágenes.")

cap.release()
