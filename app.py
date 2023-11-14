import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from collections import Counter
import os

app = Flask(__name__)
modelNB = pickle.load(open('modeln.pkl', 'rb'))
modelDC = pickle.load(open('modeld.pkl', 'rb'))
modelC4 = pickle.load(open('modeld.pkl', 'rb'))
modelRF = pickle.load(open('modelrd.pkl', 'rb'))
modelNN = pickle.load(open('modelrd.pkl', 'rb'))

bacd = {0: 88.70, 1: 90.16, 2: 76.59, 3: 89.74, 4: 89.27}
bocd = {0: 89.52, 1: 93.89, 2: 78.04, 3: 91.88, 4: 91.87}
lm = []
d = {0: "Naive Bayes", 1: "Decision Trees", 2: "C4.5", 3: "Random Forest", 4: "Neural Network"}
lm.append(modelNB)
lm.append(modelDC)
lm.append(modelDC)
lm.append(modelRF)
lm.append(modelNN)
dres = {1: "YES", 0: "NO"}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    results = []

    # Iterate through models
    for i in range(len(lm)):
        model = lm[i]
        if i == 4:
            zl = final_features
            zl = np.reshape(zl, (1, 13))

            print("a", zl)
            prediction = model.predict(zl)

            results.append(
                {'Model Name': d[i], 'Prediction': dres[round(prediction[0][0])], "BGAC": bacd[i], "BOCD": bocd[i]})

        else:
            zl = [final_features]

            print("b", zl)
            prediction = model.predict(zl)

            results.append({'Model Name': d[i], 'Prediction': dres[prediction[0]], "BGAC": bacd[i], "BOCD": bocd[i]})

    # Find the majority prediction
    predictions = [result['Prediction'] for result in results]
    majority_prediction = max(Counter(predictions), key=Counter(predictions).get)

    # Format the message with the majority prediction
    prediction_message = f"Majority Prediction: {majority_prediction}"

    return render_template('index.html', prediction_message=prediction_message)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
