
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
# Charger le modèle
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def formulaire():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_features = [np.array(features)]
    prediction = model.predict(input_features)
    return render_template('index.html', predictions = prediction)
if __name__ == '__main__':
    app.run(debug=True)


# # Récupérer les données du formulaire
    # sepal_length = float(request.form['sepal_length'])
    # sepal_width = float(request.form['sepal_width'])
    # petal_length = float(request.form['petal_length'])
    # petal_width = float(request.form['petal_width'])

    # # Préparer les données pour la prédiction
    # data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # # Faire la prédiction
    # prediction = model.predict(data)

    # # Convertir le résultat en label
    # if prediction == 0:
    #     species = 'Setosa'
    # elif prediction == 1:
    #     species = 'Versicolor'
    # else:
    #     species = 'Virginica'

    # return render_template('index.html', predictions = species)