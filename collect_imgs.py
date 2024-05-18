import os
import cv2

DATA_DIR = './data' #Creación de carpeta para almacenar capturas
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5  #Asignacion del numero de señas a capturar
dataset_size = 100  #Cantidad de fotografias a tomar para entrenar el modelo

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):

    if not os.path.exists(os.path.join(DATA_DIR, str(j))):  #Creacion de carpeta por cada seña
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j)) #Impresion en terminal el numero de seña que se esta capturando

    while True:
        #Muestra el texto de presionar Q para capturar mientras transmite el video en tiempo real
        ret, frame = cap.read()
        cv2.putText(frame, 'Presiona "Q" para capturar', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    #Una vez se presiona Q empieza a tomar fotografias de cada frame hasta que counter llege a ser igual a dataset_size
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()