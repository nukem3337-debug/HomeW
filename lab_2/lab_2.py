# 2-LABORATORIYA ISHI
# Mavzu: Pandas yordamida ma'lumotlarni yuklash, tozalash va tayyorlash

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. CSV yuklash
df = pd.read_csv("students.csv")
print("Datasetning birinchi qatorlari:")
print(df.head())

# 2. Dataset haqida ma'lumot
print("\nDataset haqida ma'lumot:")
print(df.info())
print("\nStatistik tavsif:")
print(df.describe())

# 3. Missing value aniqlash
print("\nMissing values (har bir ustunda):")
print(df.isnull().sum())

# 4. Numerik ustunlardagi bo'sh qiymatlarni o'rtacha bilan to'ldirish
df.fillna(df.mean(numeric_only=True), inplace=True)
print("\nTo'ldirilgandan keyingi missing values:")
print(df.isnull().sum())

# 5. Gender ustunini raqamlashtirish (M=0, F=1)
df["gender"] = df["gender"].map({"M": 0, "F": 1})

# 6. Modelni tayyorlash uchun X (kiruvchi) va y (natija) ajratish
X = df.drop("result", axis=1)
y = df["result"]

# 7. Train va Test guruhlarga ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain hajmi:", X_train.shape)
print("Test hajmi:", X_test.shape)

# Tozalangan datasetni saqlash
df.to_csv("students_clean.csv", index=False)
print("\nTozalangan dataset 'students_clean.csv' fayliga saqlandi.")
