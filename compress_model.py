import joblib

# Load your existing model (replace with correct filename)
model = joblib.load("swakriti_measurement_model.pkl")  # or whatever your model file is called

# Save a compressed version
joblib.dump(model, "model_compressed.pkl", compress=3)

print("✅ Compressed model saved as model_compressed.pkl")
