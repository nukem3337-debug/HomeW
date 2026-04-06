import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Grafik uchun
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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
    if row['Tarix'] >= 4.2 and row['Ona_tili'] >= 4.0:
        return 'Gumanitar'
    elif row['Matematika'] >= 4.5 and row['Chet_tili'] >= 4.0:
        return 'Biznes'
    elif row['Matematika'] >= 4.0 and row['Fizika'] >= 4.2:
        return 'IT (Raqamli)'
    elif row['Fizika'] >= 4.5:
        return 'Muhandislik'
    else:
        return 'Huquqshunoslik'

df['Yonalish'] = df.apply(yonalishni_aniqla, axis=1)

stats = df['Yonalish'].value_counts().sort_index()


plt.figure(figsize=(10, 6))
plt.plot(stats.index, stats.values, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

plt.title("Talabalarning yo'nalishlar bo'yicha taqsimoti (Line Graph)", fontsize=14)
plt.xlabel("Yo'nalishlar nomi", fontsize=12)
plt.ylabel("Talabalar soni", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)


for i, txt in enumerate(stats.values):
    plt.annotate(txt, (stats.index[i], stats.values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()


X = df[['Matematika', 'Tarix', 'Ona_tili', 'Fizika', 'Chet_tili']]
y = df['Yonalish']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Grafik ekranda ko'rsatildi. Dastur yakunlandi.")