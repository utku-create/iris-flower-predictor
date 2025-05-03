# Gerekli kütüphaneleri içe aktar
import tkinter as tk
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Iris veri setini yükle
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])  # sayıları tür ismine dönüştür

#Özellikleri (X) ve hedefi (y) ayır
X = df[iris.feature_names]
y = df['target']

#Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modeli oluştur ve eğit (K-En Yakın Komşu)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

#Test seti ile doğruluğu hesapla
y_pred = model.predict(X_test)
print("Model Doğruluğu:", accuracy_score(y_test, y_pred))

#Tür isimlerini al
species = iris.target_names

# ------------------------- #
#  TAHMİN ARAYÜZÜ (Tkinter)
# ------------------------- #

# Kullanıcıdan gelen veriye göre tahmin yapan fonksiyon
def tahmin_et():
    try:
        # Kullanıcının girdiği değerleri al
        sl = float(entry_sepal_length.get())
        sw = float(entry_sepal_width.get())
        pl = float(entry_petal_length.get())
        pw = float(entry_petal_width.get())

        # Yeni veriyle tahmin yap
        yeni_veri = [[sl, sw, pl, pw]]
        tahmin = model.predict(yeni_veri)[0]

        # Sonucu ekranda göster
        sonuc_label.config(text=f"Tahmin edilen tür: {species[tahmin]}")
    except ValueError:
        # Eğer kullanıcı geçersiz bir şey girerse uyar
        sonuc_label.config(text="Lütfen geçerli sayılar girin.")

# Tkinter penceresini oluştur
pencere = tk.Tk()
pencere.title("Iris Çiçeği Tahmin Uygulaması")

# Giriş kutuları ve etiketleri
def yeni_giris_satiri(etiket_text):
    tk.Label(pencere, text=etiket_text).pack()
    giris = tk.Entry(pencere)
    giris.pack()
    return giris

entry_sepal_length = yeni_giris_satiri("Sepal Length (cm):")
entry_sepal_width = yeni_giris_satiri("Sepal Width (cm):")
entry_petal_length = yeni_giris_satiri("Petal Length (cm):")
entry_petal_width = yeni_giris_satiri("Petal Width (cm):")

# Tahmin butonu
tk.Button(pencere, text="Tahmin Et", command=tahmin_et).pack(pady=10)

# Sonucun gösterileceği alan
sonuc_label = tk.Label(pencere, text="")
sonuc_label.pack()

# Uygulama çalışsın
pencere.mainloop()
