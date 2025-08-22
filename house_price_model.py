import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic house data for demonstration"""
        np.random.seed(42)
        
        # Generate realistic house features
        data = {
            'square_feet': np.random.normal(2000, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'distance_to_city': np.random.normal(10, 5, n_samples),
            'crime_rate': np.random.uniform(0, 100, n_samples),
            'school_rating': np.random.uniform(1, 10, n_samples)
        }
        
        # Create target variable (house price) with some noise
        base_price = (
            data['square_feet'] * 100 +  # $100 per sq ft
            data['bedrooms'] * 15000 +    # $15k per bedroom
            data['bathrooms'] * 20000 +   # $20k per bathroom
            (50 - data['age']) * 1000 +  # Newer houses cost more
            (20 - data['distance_to_city']) * 2000 +  # Closer to city = more expensive
            (100 - data['crime_rate']) * 100 +  # Lower crime = more expensive
            data['school_rating'] * 5000  # Better schools = more expensive
        )
        
        # Add some noise and ensure positive prices
        data['price'] = np.maximum(base_price + np.random.normal(0, 20000, n_samples), 50000)
        
        return pd.DataFrame(data)
    
    def prepare_data(self, data):
        """Prepare features and target for training"""
        # Select features (exclude price)
        features = ['square_feet', 'bedrooms', 'bathrooms', 'age', 
                   'distance_to_city', 'crime_rate', 'school_rating']
        
        X = data[features]
        y = data['price']
        
        self.feature_names = features
        return X, y
    
    def train_model(self, X, y):
        """Train the linear regression model"""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("Model Training Results:")
        print(f"Mean Squared Error: ${mse:,.2f}")
        print(f"Root Mean Squared Error: ${rmse:,.2f}")
        print(f"Mean Absolute Error: ${mae:,.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return X_test, y_test, y_pred, X_train_scaled, X_test_scaled
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.feature_names is None:
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        importance['abs_coefficient'] = abs(importance['coefficient'])
        importance = importance.sort_values('abs_coefficient', ascending=False)
        return importance
    
    def predict_price(self, features):
        """Predict house price for given features"""
        # Ensure model and scaler are loaded
        if self.model is None or self.scaler is None:
            try:
                self.load_model()
            except Exception as e:
                print(f"Error loading model: {e}")
                return None

        try:
            features = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def save_model(self, filename='house_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='house_price_model.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filename}")

def plot_results(y_test, y_pred, feature_importance):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price')
    axes[0, 0].set_ylabel('Predicted Price')
    axes[0, 0].set_title('Actual vs Predicted House Prices')
    
    # Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Price')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    
    # Feature Importance
    axes[1, 0].barh(feature_importance['feature'], feature_importance['coefficient'])
    axes[1, 0].set_xlabel('Coefficient Value')
    axes[1, 0].set_title('Feature Importance (Coefficients)')
    
    # Price Distribution
    axes[1, 1].hist(y_test, bins=30, alpha=0.7, label='Actual')
    axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label='Predicted')
    axes[1, 1].set_xlabel('Price')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Price Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the house price prediction"""
    print("🏠 House Price Prediction using Linear Regression")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Generate sample data
    print("\n📊 Generating sample house data...")
    data = predictor.generate_sample_data(1000)
    print(f"Generated {len(data)} sample houses")
    
    # Display sample data
    print("\n📋 Sample data:")
    print(data.head())
    print(f"\nData shape: {data.shape}")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X, y = predictor.prepare_data(data)
    
    # Train model
    print("\n🚀 Training linear regression model...")
    X_test, y_test, y_pred, X_train_scaled, X_test_scaled = predictor.train_model(X, y)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    print("\n📈 Feature Importance:")
    print(feature_importance)
    
    # Example predictions
    print("\n💡 Example Predictions:")
    sample_house = np.array([2500, 3, 2, 10, 5, 20, 8])  # [sq_ft, beds, baths, age, dist, crime, school]
    predicted_price = predictor.predict_price(sample_house)
    print(f"Sample house features: {dict(zip(predictor.feature_names, sample_house))}")
    print(f"Predicted price: ${predicted_price:,.2f}")
    
    # Save model
    predictor.save_model()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_results(y_test, y_pred, feature_importance)
    
    print("\n✅ Model training completed!")
    print("\nTo use the model for new predictions:")
    print("1. Load the saved model: predictor.load_model()")
    print("2. Prepare features: [sq_ft, beds, baths, age, dist_to_city, crime_rate, school_rating]")
    print("3. Predict: predictor.predict_price(features)")

if __name__ == "__main__":
    #execute only if run as a script
    main() 