import joblib
import numpy as np

# Load the updated model
model = joblib.load('updated_collision_risk_model.pkl')

# Sample test input (use different values)
sample_input = np.array([[36645, 29772, -0.361313, -0.393859, -0.706157, -0.334237, -0.412960, -0.512119, 0.067279]])
prediction = model.predict(sample_input)
print("Model Output:", prediction)
