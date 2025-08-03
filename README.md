# MLPoint
# 💡 MLPoint - Makine Öğrenmesi ile Veri Analizi ve Tahmin Uygulaması

KötüML, veri görselleştirme, eksik verileri tamamlama, makine öğrenmesi modeli eğitme ve eğitilmiş modellerle tahmin yapma gibi işlemleri tek bir arayüzde birleştiren Streamlit tabanlı bir AutoML uygulamasıdır.

## 🚀 Özellikler

- 📂 CSV formatında veri yükleme
- 🧹 Eksik verileri ortalama ile tamamlama
- 📊 Grafiklerle veri analizi (Histogram, Boxplot, Korelasyon, Pasta Grafiği)
- 🧠 Makine öğrenmesi modelleri ile eğitim (Regression ve Classification)
  - Linear Regression / Logistic Regression
  - Decision Tree
  - Random Forest
  - KNN
  - SVM
- 🧾 Eğitilen modeli .pkl olarak indirme
- 🔁 Eğitimli modeli yükleyerek tahmin yapma
- 📥 Tahmin sonuçlarını .csv formatında dışa aktarma

## 🧪 Kullanım

1. **Veri Yükle**: `.csv` dosyanı yükle
2. **Eksik Verileri Doldur** (isteğe bağlı)
3. **Grafikleri İncele**: Verini analiz et
4. **Model Eğit**: Hedef sütunu ve modeli seç
5. **Modeli İndir / Tahmin Yap**: Eğittiğin modeli indir ya da tekrar yükle, tahmin yap

## 🔧 Kurulum

```bash
pip install -r requirements.txt
streamlit run app.py
