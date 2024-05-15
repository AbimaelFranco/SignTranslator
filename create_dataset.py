import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Importando los módulos necesarios y estableciendo configuraciones iniciales
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializando el modelo de detección de manos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio donde se almacenarán los datos
DATA_DIR = './data'

# Listas para almacenar los datos y las etiquetas
data = []
labels = []

# Iterando sobre los directorios en el directorio de datos
for dir_ in os.listdir(DATA_DIR):
    # Iterando sobre los archivos de imagen en cada directorio
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Lista auxiliar para almacenar los datos de cada imagen
        data_aux = []

        # Listas para almacenar las coordenadas X e Y de los puntos clave de la mano
        x_ = []
        y_ = []

        # Leyendo la imagen y convirtiéndola a formato RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesando la imagen para detectar manos
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Si se detecta una mano, iterando sobre los puntos clave de la mano
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Almacenando las coordenadas X e Y de los puntos clave
                    x_.append(x)
                    y_.append(y)

                # Normalizando las coordenadas y almacenándolas en la lista auxiliar
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Agregando los datos de la imagen y la etiqueta correspondiente a las listas principales
            data.append(data_aux)
            labels.append(dir_)

# Guardando los datos y etiquetas en un archivo pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()