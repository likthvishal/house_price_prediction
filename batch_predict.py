#!/usr/bin/env python3
"""
Batch House Price Prediction Script

This script allows you to make predictions for multiple houses at once,
either from a CSV file or by manually entering multiple houses.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

def load_model(model_path='house_price_model.pkl'):
    """Load the trained model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")
        return model_data
    except FileNotFoundError:
        print(f"❌ Model file {model_path} not found!")
        print("Please train the model first using: python house_price_model.py")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_batch(model_data, houses_data):
    """Make predictions for multiple houses"""
    if not isinstance(houses_data, np.ndarray):
        houses_data = np.array(houses_data)
    
    # Scale features
    houses_scaled = model_data['scaler'].transform(houses_data)
    
    # Make predictions
    predictions = model_data['model'].predict(houses_scaled)
    
    return predictions

def load_houses_from_csv(csv_path):
    """Load house data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        required_features = ['square_feet', 'bedrooms', 'bathrooms', 'age', 
                           'distance_to_city', 'crime_rate', 'school_rating']
        
        # Check if all required features are present
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"❌ Missing required features: {missing_features}")
            print(f"Required features: {required_features}")
            return None
        
        # Extract features in correct order
        features = df[required_features].values
        
        print(f"✅ Loaded {len(features)} houses from {csv_path}")
        return features, df
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def create_sample_csv():
    """Create a sample CSV file with the correct format"""
    sample_data = {
        'square_feet': [1500, 2000, 2500, 3000, 1800],
        'bedrooms': [2, 3, 3, 4, 2],
        'bathrooms': [1, 2, 2, 3, 2],
        'age': [20, 15, 10, 5, 25],
        'distance_to_city': [12, 8, 5, 3, 15],
        'crime_rate': [40, 30, 20, 15, 50],
        'school_rating': [6, 7, 8, 9, 5]
    }
    
    df = pd.DataFrame(sample_data)
    filename = 'sample_houses.csv'
    df.to_csv(filename, index=False)
    print(f"✅ Created sample CSV file: {filename}")
    print("📋 Sample data:")
    print(df)
    return filename

def interactive_input():
    """Get house data interactively from user"""
    print("\n🏠 Interactive House Input")
    print("Enter house features (or 'done' to finish):")
    
    houses = []
    house_num = 1
    
    while True:
        print(f"\n--- House {house_num} ---")
        
        try:
            square_feet = input("Square feet (or 'done'): ").strip()
            if square_feet.lower() == 'done':
                break
            
            square_feet = float(square_feet)
            bedrooms = int(input("Bedrooms: "))
            bathrooms = int(input("Bathrooms: "))
            age = int(input("Age (years): "))
            distance_to_city = float(input("Distance to city (miles): "))
            crime_rate = float(input("Crime rate (0-100): "))
            school_rating = float(input("School rating (1-10): "))
            
            house = [square_feet, bedrooms, bathrooms, age, 
                    distance_to_city, crime_rate, school_rating]
            houses.append(house)
            house_num += 1
            
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\n\n⏹️  Input cancelled.")
            break
    
    if houses:
        houses_array = np.array(houses)
        print(f"\n✅ Input {len(houses)} houses successfully")
        return houses_array
    else:
        print("❌ No houses entered")
        return None

def display_predictions(houses_data, predictions, feature_names):
    """Display predictions in a nice format"""
    print("\n" + "="*80)
    print("🏠 HOUSE PRICE PREDICTIONS")
    print("="*80)
    
    results = []
    for i, (house, pred) in enumerate(zip(houses_data, predictions)):
        house_dict = dict(zip(feature_names, house))
        results.append({
            'House': i + 1,
            'Square Feet': f"{house_dict['square_feet']:,.0f}",
            'Bedrooms': house_dict['bedrooms'],
            'Bathrooms': house_dict['bathrooms'],
            'Age': f"{house_dict['age']} years",
            'Distance': f"{house_dict['distance_to_city']} miles",
            'Crime Rate': f"{house_dict['crime_rate']:.1f}",
            'School Rating': f"{house_dict['school_rating']:.1f}/10",
            'Predicted Price': f"${pred:,.2f}"
        })
    
    # Create DataFrame for nice display
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "-"*80)
    print("📊 PREDICTION SUMMARY")
    print("-"*80)
    print(f"Total Houses: {len(predictions)}")
    print(f"Average Price: ${np.mean(predictions):,.2f}")
    print(f"Median Price: ${np.median(predictions):,.2f}")
    print(f"Min Price: ${np.min(predictions):,.2f}")
    print(f"Max Price: ${np.max(predictions):,.2f}")
    print(f"Price Range: ${np.max(predictions) - np.min(predictions):,.2f}")

def save_results(houses_data, predictions, feature_names, filename='predictions_results.csv'):
    """Save prediction results to CSV"""
    results_data = []
    for i, (house, pred) in enumerate(zip(houses_data, predictions)):
        house_dict = dict(zip(feature_names, house))
        row = house_dict.copy()
        row['predicted_price'] = pred
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")

def main():
    """Main function"""
    print("🏠 Batch House Price Prediction")
    print("="*50)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    print(f"\n📋 Available features: {model_data['feature_names']}")
    
    # Get input method
    print("\n📥 Choose input method:")
    print("1. Load from CSV file")
    print("2. Interactive input")
    print("3. Create sample CSV file")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            csv_path = input("Enter CSV file path: ").strip()
            if not csv_path:
                print("❌ No file path provided")
                continue
                
            result = load_houses_from_csv(csv_path)
            if result:
                houses_data, original_df = result
                break
                
        elif choice == '2':
            houses_data = interactive_input()
            if houses_data is not None:
                break
                
        elif choice == '3':
            csv_path = create_sample_csv()
            print(f"\nNow you can edit {csv_path} and run option 1")
            continue
            
        elif choice == '4':
            print("👋 Goodbye!")
            return
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")
    
    if houses_data is None or len(houses_data) == 0:
        print("❌ No houses to predict")
        return
    
    # Make predictions
    print(f"\n🚀 Making predictions for {len(houses_data)} houses...")
    predictions = predict_batch(model_data, houses_data)
    
    # Display results
    display_predictions(houses_data, predictions, model_data['feature_names'])
    
    # Save results
    save_choice = input("\n💾 Save results to CSV? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes']:
        filename = input("Enter filename (default: predictions_results.csv): ").strip()
        if not filename:
            filename = 'predictions_results.csv'
        save_results(houses_data, predictions, model_data['feature_names'], filename)

if __name__ == "__main__":
    main() 