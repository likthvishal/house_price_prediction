from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Global variable to store the model
model_data = None

def load_model():
    """Load the trained model"""
    global model_data
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return True
    except FileNotFoundError:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['square_feet']),
            int(data['bedrooms']),
            int(data['bathrooms']),
            int(data['age']),
            float(data['distance_to_city']),
            float(data['crime_rate']),
            float(data['school_rating'])
        ]
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        features_scaled = model_data['scaler'].transform(features_array)
        prediction = model_data['model'].predict(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': f"${prediction:,.2f}",
            'features': dict(zip(model_data['feature_names'], features))
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    model_loaded = model_data is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # Try to load the model
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  Model not found. Please train the model first using house_price_model.py")
        print("   Run: python house_price_model.py")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 