# 4-LABORATORIYA ISHI
# Mavzu: Linear Regression yordamida prognoz qilish (Uy narxini bashorat qilish)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Datasetni yuklash
df = pd.read_csv("house.csv")
print("Dataset:")
print(df.head())

# 2. Kiruvchi (X) va chiquvchi (y) o'zgaruvchilar
X = df[["area"]]
y = df["price"]

# 3. Train-test ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Modelni yaratish va o'qitish
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Bashorat qilish
pred = model.predict(X_test)
print("\nTest uy maydonlari:", X_test["area"].tolist())
print("Bashorat narxlari:", [round(p, 2) for p in pred])

# 6. Modelni baholash
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"\nMSE: {mse:.2f}")
print(f"R2: {r2:.4f}")
print(f"Intercept (b0): {model.intercept_:.2f}")
print(f"Coefficient (b1): {model.coef_[0]:.2f}")

# 7. Grafik chizish
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Haqiqiy ma'lumotlar", color="blue")
plt.plot(X, model.predict(X), color="red", label="Regression chizig'i")
plt.xlabel("Area (m²)")
plt.ylabel("Price")
plt.title("Linear Regression — Uy narxini bashorat qilish")
plt.legend()
plt.grid(True)
plt.savefig("linear_regression.png")
plt.show()
