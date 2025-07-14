from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import logging
import joblib  
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store model and data
model = None
df = None
MEASUREMENTS_FILE = 'measurements.json'
MODEL_FILE        = 'model_compressed.pkl'

def load_data():
    """Load and prepare the data."""
    np.random.seed(42)
    n_samples = 1000
    data = {
        'age': np.random.randint(3, 18, n_samples),
        'gender': np.random.choice([1, 2], n_samples),  # 1=Male, 2=Female
        'weight': np.random.normal(40, 15, n_samples),
        'height': np.random.normal(140, 20, n_samples),
        'ethnicity': np.random.choice([1, 2, 3, 4, 5], n_samples)
    }
    df = pd.DataFrame(data)
    df = df[(df['weight'] > 0) & (df['height'] > 0)].copy()

    # Create estimated measurements based on height and other factors
    df['chest'] = df['height'] * 0.52 + (df['gender']-1.5)*2 + np.random.normal(0, 2, len(df))
    df['waist'] = df['height'] * 0.45 + (df['gender']-1.5)*1 + np.random.normal(0, 2, len(df))
    df['hip'] = df['height'] * 0.55 + (df['gender']-1.5)*3 + np.random.normal(0, 2, len(df))
    df['shoulder'] = df['height'] * 0.22 + np.random.normal(0, 1, len(df))
    df['arm_length'] = df['height'] * 0.28 + np.random.normal(0, 1, len(df))
    df['inseam'] = df['height'] * 0.45 + np.random.normal(0, 2, len(df))
    df['torso'] = df['height'] * 0.31 + np.random.normal(0, 1, len(df))
    df['neck'] = df['height'] * 0.15 + np.random.normal(0, 0.5, len(df))
    return df

def train_model(df):
    """Train the model."""
    X = df[['age', 'gender', 'weight', 'height', 'ethnicity']]
    y = df[['chest', 'waist', 'hip', 'shoulder', 'arm_length', 'inseam', 'torso', 'neck']]
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    model.fit(X, y)
    return model

def initialize_model():
    """
    Load pre‑trained model if present; otherwise train once
    and save a compressed copy for future runs.
    """
    global model, df
    if os.path.exists(MODEL_FILE):
        logger.info("Loading pre‑trained model from %s…", MODEL_FILE)
        model = joblib.load(MODEL_FILE)
        logger.info("Model loaded successfully!")
    else:
        logger.info("Model file not found; training a new model…")
        df = load_data()
        model = train_model(df)
        joblib.dump(model, MODEL_FILE, compress=3)
        logger.info("Model trained and saved to %s", MODEL_FILE)


def load_measurements():
    """Load measurements from JSON file."""
    if os.path.exists(MEASUREMENTS_FILE):
        try:
            with open(MEASUREMENTS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {MEASUREMENTS_FILE}. Starting with empty data.")
            return {}
    return {}

def save_measurements(measurements_data):
    """Save measurements to JSON file."""
    try:
        with open(MEASUREMENTS_FILE, 'w') as f:
            json.dump(measurements_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving measurements: {str(e)}")
        return False

def validate_input(data, for_update=False):
    """Validate input data."""
    required_fields = ['user_id']
    if not for_update:
        required_fields.extend(['height', 'weight', 'gender', 'age'])
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate user_id
    user_id = data['user_id']
    if not isinstance(user_id, (str, int)) or str(user_id).strip() == '':
        return False, "User ID must be a non-empty string or number"
    
    # Skip other validations for update requests with only measurements
    if for_update and len(data) == 2 and 'measurements' in data:
        return validate_measurements_format(data['measurements'])
    
    if not for_update:
        # Validate data types and ranges
        try:
            age = float(data['age'])
            if not (3 <= age <= 18):
                return False, "Age must be between 3 and 18 years"
        except (ValueError, TypeError):
            return False, "Age must be a valid number"
        
        try:
            weight = float(data['weight'])
            if not (10.0 <= weight <= 120.0):
                return False, "Weight must be between 10.0 and 120.0 kg"
        except (ValueError, TypeError):
            return False, "Weight must be a valid number"
        
        try:
            height = float(data['height'])
            if not (80.0 <= height <= 220.0):
                return False, "Height must be between 80.0 and 220.0 cm"
        except (ValueError, TypeError):
            return False, "Height must be a valid number"
        
        # Validate gender
        gender = data['gender']
        if isinstance(gender, str):
            gender_lower = gender.lower()
            if gender_lower not in ['male', 'female', 'm', 'f']:
                return False, "Gender must be 'male', 'female', 'm', or 'f'"
        elif isinstance(gender, int):
            if gender not in [1, 2]:
                return False, "Gender must be 1 (male) or 2 (female)"
        else:
            return False, "Gender must be a string or integer"
    
    return True, "Valid"

def validate_measurements_format(measurements):
    """Validate measurements format for updates."""
    if not isinstance(measurements, dict):
        return False, "Measurements must be a dictionary"
    
    valid_measurement_keys = ['chest', 'waist', 'hip', 'shoulder', 'arm_length', 'inseam', 'torso', 'neck']
    
    for key, value in measurements.items():
        if key not in valid_measurement_keys:
            return False, f"Invalid measurement key: {key}. Valid keys are: {', '.join(valid_measurement_keys)}"
        
        try:
            float_value = float(value)
            if float_value <= 0:
                return False, f"Measurement {key} must be a positive number"
        except (ValueError, TypeError):
            return False, f"Measurement {key} must be a valid number"
    
    return True, "Valid"

def convert_gender(gender):
    """Convert gender to numeric format."""
    if isinstance(gender, str):
        gender_lower = gender.lower()
        if gender_lower in ['male', 'm']:
            return 1
        elif gender_lower in ['female', 'f']:
            return 2
    elif isinstance(gender, int):
        return gender
    return 1  # Default to male

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    measurements_data = load_measurements()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'total_users': len(measurements_data),
        'measurements_file': MEASUREMENTS_FILE
    })

@app.route('/predict', methods=['POST'])
def predict_measurements():
    """Predict body measurements based on input parameters."""
    try:
        # Get JSON data from request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Prepare input data
        user_id = str(data['user_id'])
        age = float(data['age'])
        gender = convert_gender(data['gender'])
        weight = float(data['weight'])
        height = float(data['height'])
        ethnicity = 5  # Default to 'Other Race' (value 5)
        
        # Make prediction
        input_data = [[age, gender, weight, height, ethnicity]]
        prediction = model.predict(input_data)[0]
        
        # Prepare measurements
        measurements = {
            'chest': round(float(prediction[0]), 2),
            'waist': round(float(prediction[1]), 2),
            'hip': round(float(prediction[2]), 2),
            'shoulder': round(float(prediction[3]), 2),
            'arm_length': round(float(prediction[4]), 2),
            'inseam': round(float(prediction[5]), 2),
            'torso': round(float(prediction[6]), 2),
            'neck': round(float(prediction[7]), 2)
        }
        
        # Load existing measurements data
        measurements_data = load_measurements()
        
        # Save user data
        user_data = {
            'user_id': user_id,
            'input_parameters': {
                'age': age,
                'gender': 'male' if gender == 1 else 'female',
                'weight': weight,
                'height': height,
                'ethnicity': 'other'
            },
            'measurements_cm': measurements,
            'measurements_inches': {
                key: round(value / 2.54, 2) for key, value in measurements.items()
            },
            'prediction_timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'is_predicted': True,
            'is_manually_updated': False
        }
        
        measurements_data[user_id] = user_data
        
        # Save to file
        if save_measurements(measurements_data):
            logger.info(f"Measurements saved for user {user_id}")
        else:
            logger.error(f"Failed to save measurements for user {user_id}")
        
        # Prepare response
        response = {
            'success': True,
            'user_id': user_id,
            'measurements_cm': measurements,
            'measurements_inches': {
                key: round(value / 2.54, 2) for key, value in measurements.items()
            },
            'message': 'Measurements predicted and saved successfully'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/update-measurements', methods=['PUT'])
def update_measurements():
    """Update measurements for a specific user."""
    try:
        # Get JSON data from request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate input
        is_valid, message = validate_input(data, for_update=True)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        user_id = str(data['user_id'])
        
        # Load existing measurements data
        measurements_data = load_measurements()
        
        # Check if user exists
        if user_id not in measurements_data:
            return jsonify({'error': f'User {user_id} not found. Please make a prediction first.'}), 404
        
        # Get current user data
        user_data = measurements_data[user_id]
        
        # Update measurements if provided
        if 'measurements' in data:
            is_valid, message = validate_measurements_format(data['measurements'])
            if not is_valid:
                return jsonify({'error': message}), 400
            
            # Update measurements in cm
            for key, value in data['measurements'].items():
                user_data['measurements_cm'][key] = round(float(value), 2)
            
            # Update measurements in inches
            user_data['measurements_inches'] = {
                key: round(value / 2.54, 2) for key, value in user_data['measurements_cm'].items()
            }
            
            # Update metadata
            user_data['last_updated'] = datetime.now().isoformat()
            user_data['is_manually_updated'] = True
        
        # Save updated data
        if save_measurements(measurements_data):
            logger.info(f"Measurements updated for user {user_id}")
        else:
            logger.error(f"Failed to update measurements for user {user_id}")
            return jsonify({'error': 'Failed to save updated measurements'}), 500
        
        # Prepare response
        response = {
            'success': True,
            'user_id': user_id,
            'measurements_cm': user_data['measurements_cm'],
            'measurements_inches': user_data['measurements_inches'],
            'message': 'Measurements updated successfully'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error updating measurements: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get-measurements/<user_id>', methods=['GET'])
def get_measurements(user_id):
    """Get measurements for a specific user."""
    try:
        # Load measurements data
        measurements_data = load_measurements()
        
        # Check if user exists
        if user_id not in measurements_data:
            return jsonify({'error': f'User {user_id} not found'}), 404
        
        user_data = measurements_data[user_id]
        
        response = {
            'success': True,
            'user_id': user_id,
            'input_parameters': user_data.get('input_parameters', {}),
            'measurements_cm': user_data.get('measurements_cm', {}),
            'measurements_inches': user_data.get('measurements_inches', {}),
            'prediction_timestamp': user_data.get('prediction_timestamp', ''),
            'last_updated': user_data.get('last_updated', ''),
            'is_predicted': user_data.get('is_predicted', False),
            'is_manually_updated': user_data.get('is_manually_updated', False)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error retrieving measurements: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/list-users', methods=['GET'])
def list_users():
    """List all users with their basic information."""
    try:
        measurements_data = load_measurements()
        
        users_list = []
        for user_id, user_data in measurements_data.items():
            users_list.append({
                'user_id': user_id,
                'prediction_timestamp': user_data.get('prediction_timestamp', ''),
                'last_updated': user_data.get('last_updated', ''),
                'is_manually_updated': user_data.get('is_manually_updated', False)
            })
        
        return jsonify({
            'success': True,
            'total_users': len(users_list),
            'users': users_list
        })
        
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api-info', methods=['GET'])
def api_info():
    """API information endpoint."""
    return jsonify({
        'api_name': 'Swakriti Body Measurement Predictor API',
        'version': '2.0',
        'description': 'AI-powered body measurement estimation API with user management',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict body measurements',
            '/update-measurements': 'PUT - Update user measurements',
            '/get-measurements/<user_id>': 'GET - Get measurements for specific user',
            '/list-users': 'GET - List all users',
            '/api-info': 'GET - API information'
        },
        'input_formats': {
            'predict': {
                'user_id': 'string or number (required)',
                'age': 'number (3-18)',
                'gender': 'string (male/female/m/f) or number (1=male, 2=female)',
                'weight': 'number in kg (10.0-120.0)',
                'height': 'number in cm (80.0-220.0)'
            },
            'update_measurements': {
                'user_id': 'string or number (required)',
                'measurements': 'object with measurement values to update'
            }
        },
        'features': [
            'User-specific data storage',
            'Measurement predictions',
            'Manual measurement updates',
            'User data retrieval',
            'Persistent JSON storage'
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
