import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Veri Görselleştirme ve Modelleme", layout="wide")
st.title("📊 Veri Görselleştirme ve Makine Öğrenmesi Uygulaması")

with st.sidebar:
    st.markdown("## 💼 Proje: kötüML")
    st.caption("Makine öğrenmesi ve veri analizi için interaktif uygulama")

secenek = st.radio("🔍 Ne yapmak istiyorsunuz?", [
    "Veri setini yükle ve eksik verileri incele",
    "Grafikleri görselleştir",
    "Model eğit ve değerlendir"
])

if "df" not in st.session_state:
    st.session_state["df"] = None

if secenek == "Veri setini yükle ve eksik verileri incele":
    uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values=["None", "none", "NULL", "null", "NaN", "nan", ""])
        st.session_state["df"] = df
    df = st.session_state["df"]
    if df is not None:
        st.success("✅ Dosya başarıyla yüklendi!")

        max_rows = min(len(df), 100)
        num_rows = st.slider("Kaç satır görmek istersin?", min_value=5, max_value=max_rows, value=10)
        st.subheader("📋 Veri Önizlemesi")
        st.dataframe(df.head(num_rows))

        st.subheader("🧪 Eksik Verileri Doldurma")
        total_missing = df.isnull().sum().sum()
        st.write(f"Toplam eksik değer sayısı: `{int(total_missing)}`")

        if total_missing > 0:
            if st.button("Tüm sayısal sütunlardaki eksik verileri ortalama ile doldur"):
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].mean(), inplace=True)
                st.session_state["df"] = df
                st.success("Eksik veriler ortalama ile dolduruldu ✅")
                st.write(f"Kalan eksik değer sayısı: `{int(df.isnull().sum().sum())}`")

                st.subheader("📅 Güncellenmiş CSV'yi İndir")
                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📂 Güncellenmiş CSV'yi indir",
                    data=csv_download,
                    file_name="guncellenmis_veri.csv",
                    mime='text/csv'
                )
        else:
            st.info("Veride eksik değer bulunmamaktadır.")

elif secenek == "Grafikleri görselleştir":
    df = st.session_state["df"]
    if df is not None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        st.sidebar.header("📌 Grafik Ayarları")
        chart_type = st.sidebar.selectbox("Grafik türünü seçin", ["Histogram", "Boxplot", "Korelasyon Isı Haritası", "Pie Chart"])

        fig = None
        if chart_type == "Histogram":
            column = st.sidebar.selectbox("Sütun seçin", numeric_columns)
            fig = px.histogram(df, x=column)
            st.plotly_chart(fig)

        elif chart_type == "Boxplot":
            column = st.sidebar.selectbox("Sütun seçin", numeric_columns)
            fig = px.box(df, y=column)
            st.plotly_chart(fig)

        elif chart_type == "Korelasyon Isı Haritası":
            st.subheader("📈 Korelasyon Matrisi")
            fig_, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig_)

        elif chart_type == "Pie Chart":
            if categorical_columns:
                column = st.sidebar.selectbox("Kategorik sütun seçin", categorical_columns)
                value_counts = df[column].value_counts().reset_index()
                value_counts.columns = [column, "Count"]
                fig = px.pie(value_counts, names=column, values="Count", hole=0.4)
                st.plotly_chart(fig)
            else:
                st.warning("Uygun kategorik sütun bulunamadı.")

elif secenek == "Model eğit ve değerlendir":
    st.header("🎯 Makine Öğrenmesi: Model Eğitimi")
    df = st.session_state["df"]
    if df is not None:
        target = st.selectbox("🌟 Hedef değişkeni seçin (label)", df.columns)
        features = st.multiselect("🧠 Özellik sütunlarını seçin (feature)", df.columns.drop(target))

        if features:
            if df[target].dtype in ['float64', 'int64'] and df[target].nunique() > 10:
                problem_type = "Regresyon"
                model_dict = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor()
                }
            else:
                problem_type = "Sınıflandırma"
                model_dict = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier()
                }

            st.markdown(f"**Algılanan Problem Türü:** :green[{problem_type}]")

            model_names = list(model_dict.keys())
            selected_model_name = st.selectbox("Model seçin", model_names)
            selected_model = model_dict[selected_model_name]

            if st.button("🚀 Modeli Eğit"):
                try:
                    X = df[features]
                    y = df[target]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    selected_model.fit(X_train, y_train)
                    y_pred = selected_model.predict(X_test)

                    st.success("✅ Model eğitildi! Artık tahmin için hazır.")

                    st.markdown("### 🔢 Kullanılan Özellikler")
                    st.code(", ".join(X_train.columns), language="markdown")

                    if problem_type == "Regresyon":
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.markdown(f"📉 Ortalama Kare Hata (MSE): `{mse:.4f}`")
                        st.markdown(f"📈 R² Skoru: `{r2:.4f}`")

                    else:
                        acc = accuracy_score(y_test, y_pred)
                        st.markdown(f"📊 Doğruluk (Accuracy): `{acc:.4f}`")
                        st.text("Sınıflandırma Raporu:")
                        st.text(classification_report(y_test, y_pred))
                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")
        else:
            st.warning("⚠️ Lütfen en az bir özellik (feature) seçin.")
