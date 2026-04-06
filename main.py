import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

data = {
    'Matematika': np.random.uniform(3.0, 5.0, 100).round(1),
    'Tarix': np.random.uniform(3.0, 5.0, 100).round(1),
    'Ona_tili': np.random.uniform(3.0, 5.0, 100).round(1),
    'Fizika': np.random.uniform(3.0, 5.0, 100).round(1),
    'Chet_tili': np.random.uniform(3.0, 5.0, 100).round(1),
}
df = pd.DataFrame(data)

def yonalishni_aniqla(row):
    if row['Tarix'] >= 4.0 and row['Ona_tili'] >= 4.0 and row['Chet_tili'] >= 4.0:
        return 'Gumanitar'
    elif row['Matematika'] >= 4.0 and row['Chet_tili'] >= 4.0:
        return 'Biznes'
    elif row['Matematika'] >= 4.2 and row['Fizika'] >= 4.0:
        return 'Raqamli texnologiyalar'
    elif row['Fizika'] >= 4.2:
        return 'Muhandislik'
    else:
        return 'Huquqshunoslik'


df['Yonalish'] = df.apply(yonalishni_aniqla, axis=1)

print("--- O'qitish uchun tayyorlangan ma'lumotlardan namuna ---")
print(df.head(), "\n")

X = df[['Matematika', 'Tarix', 'Ona_tili', 'Fizika', 'Chet_tili']]
y = df['Yonalish']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
aniqlik = accuracy_score(y_test, y_pred)
print(f"Modelning aniqlik darajasi: {aniqlik * 100:.2f}%\n")

yangi_oquvchi_baholari = [[4.8, 3.5, 3.8, 4.6, 4.0]]

bashorat = model.predict(yangi_oquvchi_baholari)

print("--- Yangi o'quvchi natijasi ---")
print(f"Kiritilgan baholar: {yangi_oquvchi_baholari[0]}")
print(f"Tavsiya etilgan kasb yo'nalishi: {bashorat[0].upper()}")