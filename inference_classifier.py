import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import os
import pyttsx3

start_time = None
current_letter = None
progress_bar_width = 0
nombre_archivo = "Texto.txt"
contador_espacios = 0
tiempo_deteccion = 2

with open(nombre_archivo, "w") as archivo:
    pass

# Define función para reiniciar la barra de progreso
def reset_progress():
    global start_time, progress_bar_width
    start_time = time.time()
    progress_bar_width = 0

def texto_audio():
    with open(nombre_archivo, "r") as file:
        textoObtenido = file.read()

    tts = gTTS(text=textoObtenido, lang='es')
    tts.save("audio.mp3")
    os.system("start audio.mp3")

def texto_audio2():
    with open(nombre_archivo, "r") as file:
        textoObtenido = file.readlines()
    
    engine = pyttsx3.init()
    
    for line in textoObtenido:
        engine.say(line)
        engine.runAndWait()


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

diccionario = {}

diccionario[0] = 'hola'
diccionario[1] = 'a'
diccionario[2] = 'b'
diccionario[3] = 'c'
diccionario[4] = 'd'
diccionario[5] = 'e'
diccionario[6] = 'f'
diccionario[7] = 'g'
diccionario[8] = 'h'
diccionario[9] = 'i'
diccionario[10] = 'j'
diccionario[11] = 'k'
diccionario[12] = 'l'
diccionario[13] = 'm'
diccionario[14] = 'n'
diccionario[15] = 'ñ'
diccionario[16] = 'o'
diccionario[17] = 'p'
diccionario[18] = 'q'
diccionario[19] = 'r'
diccionario[20] = 's'
diccionario[21] = 't'
diccionario[22] = 'u'
diccionario[23] = 'v'
diccionario[24] = 'w'
diccionario[25] = 'x'
diccionario[26] = 'y'
diccionario[27] = 'z'
# PALABRAS
diccionario[28] = 'yo'
diccionario[29] = 'tu'
diccionario[30] = 'el/ella'
diccionario[31] = 'caminar'
diccionario[32] = 'comer'
diccionario[33] = 'dormir'
diccionario[34] = 'tomar'
diccionario[35] = 'agua'
diccionario[36] = 'cafe'
diccionario[37] = 'cerveza'
diccionario[38] = 'gaseosa'
diccionario[39] = 'bañar'
diccionario[40] = 'atencion'
diccionario[41] = 'aqui'
diccionario[42] = 'alla'
diccionario[43] = 'arriba'
diccionario[44] = 'ayer'
diccionario[45] = 'mañana'
diccionario[46] = 'besar'
diccionario[47] = 'desocupado'
diccionario[48] = 'solo'
diccionario[49] = 'ingeniero'
diccionario[50] = 'nan'


# labels_dict = { 0: 'Hola', 1: 'Yo', 
#                 2: 'Tu', 3: 'Caminar', 
#                 4: 'NAN'}

# diccionario[0] = 'hola'
# diccionario[1] = 'yo'
# diccionario[2] = 'tú'
# diccionario[3] = 'caminar'
# diccionario[4] = 'a'
# diccionario[5] = 'e'
# diccionario[6] = 'i'
# diccionario[7] = 'o'
# diccionario[8] = 'u'
# diccionario[9] = 'nan'

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        contador_espacios = 0
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if len(data_aux) <= 42: 
            # print(type(data_aux))
            prediction = model.predict([np.asarray(data_aux)])

            try:
                predicted_character = diccionario[int(prediction[0])]
            except:
                predicted_character = diccionario[50]

        if predicted_character == current_letter:
            if start_time is None:
                start_time = time.time()
            else:
                elapsed_time = time.time() - start_time
                progress_bar_width = int((elapsed_time / tiempo_deteccion) * 100)  #Tiempo de deteccion de seña
                longitud = int(progress_bar_width/100 * (x2-x1))
                progress = "Progreso: " + str(progress_bar_width) + "%"
                print(progress)
                progress_bar_width = min(progress_bar_width, 100)  # Limitar al 100%
                
                # print(str(x1 + longitud))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)    #Cuadro
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)  #Letra 
                cv2.rectangle(frame, (x1, y2 + 5), (x1 + longitud, y2 + 15), (0, 255, 0), -1)    #Barra de progreso
                cv2.putText(frame, str(progress), (x1, y2 +30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)    #Texto de %

                
                if elapsed_time >= tiempo_deteccion:
                    # Guardar la letra
                    cv2.putText(frame, "Hecho", (0+5, H-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)  #Letra 
                    print("Señal detectada:", current_letter)
                    with open(nombre_archivo, "a") as archivo1:
                        archivo1.write(current_letter)
                    reset_progress()
        else:
            # Cambiar la letra y reiniciar el progreso
            current_letter = predicted_character
            reset_progress()

    else:   #Si no detecta ninguna mano
        # cv2.line(frame, (0, H - 1), (W - 1, H - 1), (0, 255, 0), 2)
        if start_time is None and contador_espacios < 2 :
                start_time = time.time()
        elif contador_espacios == 0:
                elapsed_time = time.time() - start_time
                progress_bar_width = int((elapsed_time / 3) * 100)
                longitud = int(progress_bar_width/100 * (W))
                progress_bar_width = min(progress_bar_width, 100)  # Limitar al 100%
                cv2.rectangle(frame, (0, H - 10), (longitud, H-1), (0, 255, 0), -1)    #Barra de progreso
                if elapsed_time >= 3:
                    texto = " "
                    with open(nombre_archivo, "a") as archivo1:
                        archivo1.write(texto)
                    reset_progress()
                    contador_espacios += 1
        elif contador_espacios == 1:
                elapsed_time = time.time() - start_time
                progress_bar_width = int((elapsed_time / 3) * 100)
                longitud = int(progress_bar_width/100 * (W))
                progress_bar_width = min(progress_bar_width, 100)  # Limitar al 100%
                cv2.rectangle(frame, (0, H - 10), (longitud, H-1), (255, 0, 0), -1)    #Barra de progreso
                if elapsed_time >= 3:
                    texto = "."
                    with open(nombre_archivo, "a") as archivo1:
                        archivo1.write(texto)
                    reset_progress()     
                    contador_espacios += 1 
        elif contador_espacios == 2:
                elapsed_time = time.time() - start_time
                progress_bar_width = int((elapsed_time / 3) * 100)
                longitud = int(progress_bar_width/100 * (W))
                progress_bar_width = min(progress_bar_width, 100)  # Limitar al 100%
                cv2.rectangle(frame, (0, H - 10), (longitud, H-1), (0, 0, 255), -1)    #Barra de progreso
                if elapsed_time >= 3:
                    break

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
texto_audio2()