from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

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
        county = data['county']
        
        ##### Using date and county, make predicition here
        
        pred = 5.5
        
        
        # Placeholder for your ML model's prediction logic
        prediction_result = f"Snow Day Probability: {pred} for {date} in {county}"
        return jsonify({'prediction': prediction_result})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
