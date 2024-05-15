import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carga los datos desde el archivo pickle "data.pickle" y los almacena en un diccionario
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convierte los datos y etiquetas del diccionario en arreglos numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divide los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializa un clasificador de bosque aleatorio
model = RandomForestClassifier()

# Entrena el modelo utilizando los datos de entrenamiento
model.fit(x_train, y_train)

# Realiza predicciones sobre los datos de prueba
y_predict = model.predict(x_test)

# Calcula la precisión del modelo comparando las predicciones con las etiquetas reales
score = accuracy_score(y_predict, y_test)

# Imprime el porcentaje de muestras clasificadas correctamente
print('{}% of samples were classified correctly !'.format(score * 100))

# Guarda el modelo entrenado en un archivo pickle llamado "model.p"
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()