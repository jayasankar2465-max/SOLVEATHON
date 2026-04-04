import sqlite3
import pandas as pd
import joblib

# connect to your database
conn = sqlite3.connect("mess.db")

# read your table (change table name if needed)
df = pd.read_sql_query("SELECT * FROM feedbacks", conn)

df['rating'] = df['s'].apply(lambda x: x.count("★"))

df = df.drop(columns=['c', 'user_id', 'is_anonymous', 'm'])
df = df.drop(columns=['id'], errors='ignore')
df = pd.get_dummies(df, columns=[
    'target_item',
    'meal_time',
    'block',
    'issues'
])

X = df.drop(columns=['wastage', 's'])  # inputs
y = df['wastage']                      # what we predict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


joblib.dump(X.columns.tolist(), "model_columns.pkl")
model = RandomForestRegressor()
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

preds = model.predict(X_test)

print("Error:", mean_absolute_error(y_test, preds))

joblib.dump(model, "waste_model.pkl")

