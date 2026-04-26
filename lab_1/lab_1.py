# 1-LABORATORIYA ISHI
# Mavzu: NumPy asoslari va matritsalar bilan ishlash
# Maqsad: 10 ta talabaning 5 ta fan bo'yicha ballarini tahlil qilish

import numpy as np

# 10 ta talaba, 5 ta fan uchun tasodifiy ballar (50-99 oraliqda)
np.random.seed(42)
grades = np.random.randint(50, 100, (10, 5))

print("Talabalar ballari (10 ta talaba, 5 ta fan):")
print(grades)

# Statistik hisob-kitob
print("\nEng yuqori ball:", grades.max())
print("Eng past ball:", grades.min())
print("Umumiy o'rtacha:", round(grades.mean(), 2))

# Har bir talabaning o'rtacha bali
student_average = grades.mean(axis=1)
print("\nHar bir talabaning o'rtacha bali:")
for i, avg in enumerate(student_average):
    print(f"  Talaba {i+1}: {avg:.2f}")

# Eng yaxshi talabani aniqlash
best_student = np.argmax(student_average)
print(f"\nEng yaxshi talaba: Talaba {best_student + 1} "
      f"(o'rtacha {student_average[best_student]:.2f})")

# Natijani CSV faylga saqlash
np.savetxt("students_grades.csv", grades, delimiter=",", fmt='%d')
print("\nNatija 'students_grades.csv' fayliga saqlandi.")
