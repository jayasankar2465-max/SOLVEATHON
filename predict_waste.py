import joblib
import pandas as pd


model = joblib.load("waste_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# example new input

data = pd.DataFrame([{
    "rating": 2,
    "target_item_Chapathi": 1,
    "meal_time_Dinner": 1,
    "block_A": 1,
    "issues_Texture": 1
}])
data = data.reindex(columns=model_columns, fill_value=0)

prediction = model.predict(data)

print("Predicted waste:", prediction[0])

prediction = prediction[0]

if prediction > 40:
    level = "High Waste Risk"
elif prediction > 20:
    level = "Moderate Waste Risk"
else:
    level = "Low Waste Risk"

print(level)