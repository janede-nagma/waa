import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error
df = pd.read_csv("Synthetic_Graduate_Admissions.csv")
X=df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y=df['Chance of Admit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Machine": SVR(),
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    mse=mean_squared_error(y_test, y_pred)
    print(f"{name} R² Score: {score:.4f} MSE: {mse:.2f}")

sample=[[320,110,4,4.5,4.0,9.0,1]]
best_model=RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
prediction=best_model.predict(sample)
print(f"Predicted Chance of Admit: {prediction[0]:.4f}")
