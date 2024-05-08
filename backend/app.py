from flask import Flask, jsonify, request
from flask_cors import CORS
from model.model import *


app = Flask(__name__)
CORS(app)
snow_model = SnowDayModel()
snow_model.load_model('model/model.pkl')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Options response
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return response
    elif request.method == 'POST':
        # Actual prediction
        data = request.get_json()
        date = data['date']
        
        ##### Using date and county, make predicition here
        
        
        pred = snow_model.predict(date)[0]
        print(pred)
        
        if(pred == -1):
            prediction_result = "N/A. Please choose a present or past date, not a future date."
        else:
            prediction_result = f"There is a {100*pred}% chance of a Snow Day for {date} in Howard County"
            
            
        return jsonify({'prediction': prediction_result})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
