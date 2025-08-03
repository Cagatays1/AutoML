# MLPoint GÜNCELLENMİŞ ARAYÜZ (Streamlit)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="MLPoint - Akıllı ML Aracı", layout="wide")
st.title("🤖 MLPoint: Akıllı Makine Öğrenmesi Yardımcısı")

with st.sidebar:
    st.markdown("## 🚀 Proje: MLPoint")
    st.caption("Bir makine öğrenmesi ve veri analiz platformu.")

# Global veri saklama
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = None

# Ana işlem menüsü
st.header("👋 Hoş geldiniz!")
secenek = st.selectbox("Ne yapmak istiyorsunuz?", (
    "Veri seti yükle ve eksik verileri doldur",
    "Veriyi analiz et ve grafikleri incele",
    "Model eğit",
    "Tahmin yap (eğitilmiş modeli kullan)"
))

# 1. Veri Yükleme ve Eksik Veri Doldurma
if secenek == "Veri seti yükle ve eksik verileri doldur":
    uploaded_file = st.file_uploader("CSV dosyası yükleyin", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Veri yüklendi!")
        st.dataframe(df.head())

        st.subheader("🔍 Eksik Verileri İncele")
        st.write(df.isnull().sum())

        if st.button("Tüm sayısal eksikleri ortalama ile doldur"):
            df = df.fillna(df.mean(numeric_only=True))
            st.session_state.df = df
            st.success("Eksik değerler dolduruldu!")

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Doldurulmuş CSV'yi indir", csv, file_name="doldurulmus_veri.csv", mime='text/csv')

# 2. Grafik İnceleme
elif secenek == "Veriyi analiz et ve grafikleri incele":
    df = st.session_state.df
    if df is not None:
        st.subheader("📊 Korelasyon Matrisi")
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)

        st.subheader("📈 Histogram")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_col = st.selectbox("Histogram için sütun seç", num_cols)
        fig_hist = px.histogram(df, x=selected_col)
        st.plotly_chart(fig_hist)

        st.subheader("📊 Kategorik Veri Pasta Grafiği")
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            selected_cat = st.selectbox("Pasta grafik için sütun seç", cat_cols)
            fig_pie = px.pie(df, names=selected_cat)
            st.plotly_chart(fig_pie)
        else:
            st.info("Kategorik veri sütunu bulunamadı.")

# 3. Model Eğitimi
elif secenek == "Model eğit":
    df = st.session_state.df
    if df is not None:
        st.subheader("🎯 Hedef ve Özellik Seçimi")
        target = st.selectbox("Hedef sütunu seçin", df.columns)
        features = st.multiselect("Özellik sütunlarını seçin", df.columns.drop(target))

        if features:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Problem tipi belirleme
            if y.dtype in ['float64', 'int64'] and y.nunique() > 10:
                problem_type = "regression"
                model_options = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "SVR": SVR()
                }
            else:
                problem_type = "classification"
                model_options = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "SVM": SVC(),
                    "Naive Bayes": GaussianNB()
                }

            st.markdown(f"### 🤖 Önerilen modeller: {', '.join(model_options.keys())}")

            selected_model_name = st.selectbox("Model seçin", list(model_options.keys()))
            model = model_options[selected_model_name]

            if st.button("🚀 Modeli Eğit"):
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.session_state.features = features

                y_pred = model.predict(X_test)
                if problem_type == "regression":
                    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.4f}")
                    st.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
                else:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    st.text(classification_report(y_test, y_pred))

                joblib.dump({"model": model, "features": features}, "egitilmis_model.pkl")
                with open("egitilmis_model.pkl", "rb") as f:
                    st.download_button("📦 Eğitilen Modeli İndir (.pkl)", f, file_name="model.pkl")

# 4. Tahmin
elif secenek == "Tahmin yap (eğitilmiş modeli kullan)":
    uploaded_model = st.file_uploader("Eğitilmiş model dosyasını (.pkl) yükleyin", type="pkl")
    if uploaded_model:
        model_data = joblib.load(uploaded_model)
        model = model_data["model"]
        features = model_data["features"]

        st.session_state.model = model
        st.session_state.features = features

        st.success("✅ Model başarıyla yüklendi!")

    model = st.session_state.get("model")
    features = st.session_state.get("features")

    if model and features:
        st.subheader("📝 Tahmin için giriş değerleri")
        user_input = {}
        for feat in features:
            user_input[feat] = st.number_input(f"{feat}", step=0.01)
        input_df = pd.DataFrame([user_input])

        if st.button("📈 Tahmin Yap"):
            prediction = model.predict(input_df)[0]
            readable_result = "✅ İçilebilir" if prediction == 1 else "🚫 İçilemez"
            st.success(f"🔮 Tahmin Sonucu: {prediction}({readable_result})")

            st.download_button("📄 Bu tahmini indir (.csv)",
                               input_df.assign(Prediction=prediction).to_csv(index=False),
                               file_name="tahmin_sonucu.csv", mime="text/csv")
    else:
        st.warning("Eğitilmiş model bulunamadı. Lütfen önce model yükleyin veya eğitin.")


