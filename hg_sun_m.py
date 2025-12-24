# Hangi conda env’deyim?
#   conda env list
#
# Env içindeki paketleri gör:
#   conda list
#
# Base env’yi .yaml olarak dışarı aktar (proje taşımak için iyi):
#   conda env export > environment.yaml
#
# Yeni env oluştur:
#   conda create -n acm4762025_env
#
# Env aktif et:
#   conda activate acm4762025_env
#
# YAML dosyasından env kur:
#   conda env create -f environment.yaml
#
# Paketleri toplu güncelleme (bazen uyumsuzluk patlatır):
#   conda update --all
#
# Option + Shift + E


# 1) IMPORTS (başta hepsini topladım ki dosya düzenli olsun)

# Temel veri bilimi
import os
import numpy as np
import pandas as pd

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# HF dataset
from datasets import load_dataset

# Sklearn - genel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# Sklearn - metin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Clustering / PCA / Regression (week4 tarafı)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Model kaydetme
import joblib

# (Opsiyonel) Yellowbrick ile elbow görselleştirme istersen:
# pip install yellowbrick
try:
    from yellowbrick.cluster import KElbowVisualizer
    YELLOWBRICK_AVAILABLE = True
except Exception:
    YELLOWBRICK_AVAILABLE = False

# ============================================================
# 2) ORTAK YARDIMCI FONKSİYONLAR (def’leri başta tanımladım)
# ============================================================

def num_summary(dataframe: pd.DataFrame, numerical_col: str, plot: bool = True, bins: int = 20):
    # Bu fonksiyon sayısal bir değişkenin:
    # - describe() istatistiklerini
    # - quantile değerlerini
    # - istenirse histogramını verir
    print("-----------", numerical_col, "-----------")
    print(dataframe[numerical_col].describe())
    print("\nQuantiles:")
    print(dataframe[numerical_col].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).T)

    if plot:
        dataframe[numerical_col].hist(bins=bins)
        plt.xlabel(numerical_col)
        plt.title(f"Distribution of {numerical_col}")
        plt.show()

def target_summary_with_num(dataframe: pd.DataFrame, target: str, numerical_col: str):
    # Target değişkenine göre (0/1 gibi) sayısal kolonun ortalamasını verir
    print("-----------", numerical_col, "-----------")
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

def compute_high_corr_columns(df_numeric: pd.DataFrame, corr_threshold: float = 0.90):
    # Korelasyonu yüksek kolonları bulmak için pratik bir yöntem:
    # - abs korelasyon matrisi
    # - üst üçgen (diagonal hariç) taranır
    # - eşik üzeri kolonlar drop list’e girer
    corr_matrix = df_numeric.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_list = [col for col in upper_triangle.columns if any(upper_triangle[col] > corr_threshold)]
    return drop_list, corr_matrix

def plot_confusion_matrix(cm: np.ndarray, labels=("0", "1"), title="Confusion Matrix"):
    # Basit bir confusion matrix görseli
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

# ============================================================
# 3) (A) HUGGINGFACE: TURKISH OFFENSIVE LANGUAGE DETECTION
# ============================================================

def run_hf_offensive_language_pipeline(
    hf_dataset_name: str = "Toygar/turkish-offensive-language-detection",
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
    model_out_dir: str = "outputs_hf_offensive",
):
    # Bu fonksiyon HuggingFace üzerinden dataset’i çekip:
    # 1) EDA (label dağılımı, tweet uzunluğu, kelime sayısı)
    # 2) TF-IDF + Logistic Regression ile sınıflandırma
    # 3) Temel metrikler + confusion matrix
    # 4) Modeli + vectorizer’ı + scaler (varsa) kaydetme
    #
    # Not: Burada metin sınıflandırma yaptığımız için StandardScaler genelde TF-IDF için kullanılmaz.
    # Ama senin ilk kodunda “tweet_length, word_count” için StandardScaler örneği vardı.
    # Ben onu “ek bir özellik mühendisliği örneği” olarak gösteriyorum.

    os.makedirs(model_out_dir, exist_ok=True)

    # -----------------------------
    # Dataset’i HF’den yükleme
    # -----------------------------
    dataset = load_dataset(hf_dataset_name)

    # HF dataset -> pandas
    # Genelde dataset["train"] içinde text ve label oluyor
    df = dataset["train"].to_pandas()

    print("\n[HF] İlk 5 satır:")
    print(df.head())
    print(f"[HF] Toplam satır sayısı: {len(df)}")

    # Genel bilgi
    print("\n[HF] df.info():")
    print(df.info())

    # Eksik veri kontrol
    print("\n[HF] Missing values:")
    print(df.isnull().sum())

    # Label dağılımı
    print("\n[HF] Label counts:")
    print(df[label_col].value_counts())
    print("\n[HF] Label percentage:")
    print(df[label_col].value_counts(normalize=True) * 100)

    # -----------------------------
    # Basit metin uzunluğu özellikleri
    # -----------------------------
    df["tweet_length"] = df[text_col].astype(str).apply(len)  # karakter sayısı
    df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))  # kelime sayısı

    print("\n[HF] Tweet length stats:")
    print(df["tweet_length"].describe())

    plt.hist(df["tweet_length"], bins=50)
    plt.title("Tweet Length Distribution")
    plt.xlabel("Character count")
    plt.ylabel("Frequency")
    plt.show()

    print("\n[HF] Word count stats:")
    print(df["word_count"].describe())

    # En uzun / en kısa tweet (kelime sayısına göre)
    # idxmax/idxmin NaN olursa patlayabilir, o yüzden text’i string’e çevirdiğimizi varsayıyorum.
    print("\n[HF] En uzun tweet (word_count max):")
    print(df.loc[df["word_count"].idxmax()])

    print("\n[HF] En kısa tweet (word_count min):")
    print(df.loc[df["word_count"].idxmin()])

    # Saldırgan vs temiz ortalama uzunluk
    print("\n[HF] Ortalama tweet_length (label bazlı):")
    print(df.groupby(label_col)["tweet_length"].mean())

    # Boxplot: kelime sayısı dağılımı
    sns.boxplot(x=df["word_count"], color="orange")
    plt.title("Kelime Sayısı Dağılımı")
    plt.show()

    # -----------------------------
    # Ufak “not defteri” örneği:
    # label ortalaması üstü filtre (senin notun)
    # -----------------------------
    # label 0/1 olduğu için mean genelde 0.xx olur, >= mean demek çoğu zaman label==1 seçer.
    df[label_col] = df[label_col].astype(int)
    mean_label = df[label_col].mean()
    filtered_df = df.loc[df[label_col] >= mean_label]
    print("\n[HF] label mean:", mean_label)
    print("[HF] mean üstü filtrelenmiş örnek:")
    print(filtered_df.head())

    # -----------------------------
    # Korelasyon işleri (bu dataset’te sayısal kolonlar az)
    # -----------------------------
    # Sayısal kolonları seçip label hariç target summary yapalım
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"] and col != label_col]
    for col in num_cols:
        target_summary_with_num(df, label_col, col)

    # Korelasyon matrisi
    corr = df.corr(numeric_only=True)
    print("\n[HF] Corr matrix:")
    print(corr)

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Korelasyon Isı Haritası (HF)")
    plt.show()

    # Yüksek korelasyonlu değişkenleri bulup drop etme
    # Burada genelde tweet_length ve word_count arasında korelasyon çıkabilir.
    df_numeric = df[[c for c in df.columns if df[c].dtype in ["int64", "float64"]]]
    high_corr_cols, corr_abs = compute_high_corr_columns(df_numeric, corr_threshold=0.90)

    print("\n[HF] Silinecek yüksek korelasyonlu değişkenler (>=0.90):")
    print(high_corr_cols)

    df_reduced = df.drop(columns=high_corr_cols, errors="ignore")

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_abs, cmap="coolwarm", annot=False)
    plt.title("Orijinal Korelasyon Matrisi (abs)")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_reduced.corr(numeric_only=True).abs(), cmap="coolwarm", annot=False)
    plt.title("Reduced Frame Korelasyon Matrisi (abs)")
    plt.show()

    # -----------------------------
    # (Metin Modeli) TF-IDF + Logistic Regression
    # -----------------------------
    # Burada asıl “metin sınıflandırma” kısmı başlıyor:
    # - Metni TF-IDF’e çeviriyoruz (kelime frekansı + ağırlık)
    # - Logistic Regression ile 0/1 sınıflandırıyoruz
    #
    # Not: Türkçe için bazen stopwords, stemming vs. eklenir.
    # Ama şu an baseline model yeterli.

    X_text = df[text_col].astype(str).values
    y = df[label_col].astype(int).values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # TF-IDF ayarları:
    # - ngram_range=(1,2) demek: unigram + bigram kullan.
    # - max_features: çok büyük olmasın diye limit koyabiliriz.
    # - min_df: çok nadir kelimeleri eleyebilir.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Logistic Regression:
    # - solver: liblinear küçük/orta veri için pratik
    # - class_weight="balanced": dengesiz label varsa yardımcı olabilir
    base_model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced"
    )

    # (Opsiyonel) GridSearchCV ile basit hyperparameter araması:
    # Bu adım biraz zaman alır, kapatmak istersen direkt base_model.fit kullan.
    param_grid = {
        "C": [0.1, 1, 3, 10],
        "penalty": ["l1", "l2"]
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train_tfidf, y_train)
    best_model = grid.best_estimator_

    print("\n[HF] Best params:", grid.best_params_)

    # Tahmin
    y_pred = best_model.predict(X_test_tfidf)

    # Metrikler
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n[HF] Metrics:")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)

    print("\n[HF] Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("\n[HF] Confusion Matrix:\n", cm)
    plot_confusion_matrix(cm, labels=("clean(0)", "offensive(1)"), title="HF Offensive - Confusion Matrix")

    # -----------------------------
    # Ek not: StandardScaler örneği (sayısal feature’lar için)
    # -----------------------------
    # Metin TF-IDF zaten sparse matris; scaler burada mantıklı değil.
    # Ama tweet_length ve word_count’u “sayısal özellik” olarak standardize etmeyi göstermek için:
    X_numeric = df[["tweet_length", "word_count"]].copy()
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=X_numeric.columns)
    print("\n[HF] Scaled numeric features head():")
    print(X_numeric_scaled_df.head())

    # Bu numeric özellikleri TF-IDF ile birleştirmek istersen (ileri seviye):
    # - scipy.sparse.hstack ile birleştirilir
    # -----------------------------
    # Modeli ve vectorizer’ı kaydetme
    # -----------------------------
    joblib.dump(best_model, os.path.join(model_out_dir, "logreg_tfidf_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_out_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(model_out_dir, "numeric_scaler.pkl"))

    print(f"\n[HF] Kaydedildi: {model_out_dir}/logreg_tfidf_model.pkl")
    print(f"[HF] Kaydedildi: {model_out_dir}/tfidf_vectorizer.pkl")
    print(f"[HF] Kaydedildi: {model_out_dir}/numeric_scaler.pkl")

    return {
        "df": df,
        "model": best_model,
        "vectorizer": vectorizer,
        "scaler": scaler
    }

# ============================================================
# 4) (B) WEEK4 TARZI: SENSÖR VERİSİ (CSV -> EDA -> KMeans -> PCA -> Regression)
# ============================================================

def run_week4_sensor_pipeline(
    csv_path: str,
    target_col: str = "Target",
    timestamp_col: str = "Timestamp",
    corr_threshold: float = 0.90,
    random_state: int = 17,
    out_dir: str = "outputs_week4_sensor",
):
    # 1) CSV oku
    # 2) df.info, null kontrol, describe
    # 3) Target dağılım çarpıklığı (mean üstü/altı count)
    # 4) Sensör kolonlarını seç (timestamp ve target dışındaki sayısal kolonlar)
    # 5) Tek tek boxplot/hist örnekleri
    # 6) Target’a göre sensör ortalamaları
    # 7) Sensör korelasyonu + yüksek korelasyon drop
    # 8) StandardScaler
    # 9) Elbow ile optimum k (varsa yellowbrick)
    # 10) KMeans cluster label ekleme
    # 11) PCA ile explained variance
    # 12) (Opsiyonel) Regression (LinearRegression) + örnek export

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # CSV okuma
    # -----------------------------
    # Burada path’i sen vereceksin:
    # ör: "/Users/.../data_ready_for_cs.csv"
    df = pd.read_csv(csv_path)

    print("\n[WEEK4] İlk 5 satır (kontrol):")
    print(df.head())

    print("\n[WEEK4] df.shape:", df.shape)

    print("\n[WEEK4] df.info():")
    print(df.info())

    print("\n[WEEK4] Missing value kontrol (any):", df.isnull().values.any())
    print("[WEEK4] Missing value count (sum):\n", df.isnull().sum())

    # -----------------------------
    # Descriptive stats
    # -----------------------------
    # Sadece sayısal kolonları alalım:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n[WEEK4] Numeric columns:", len(numeric_cols))

    desc_summ = df[numeric_cols].describe().T
    print("\n[WEEK4] df.describe().T:")
    print(desc_summ)

    # Numpy array’e çevirme notu:
    df_desc_na = np.array(desc_summ)
    df_na = np.array(df)  # tüm df’yi array yapar
    # Bu iki satır “neden var?” -> bazı numpy vektör işlemlerinde DataFrame yerine array pratik olur.

    # -----------------------------
    # Target analizi
    # -----------------------------
    if target_col in df.columns:
        target_mean = df[target_col].mean()
        above_count = df[df[target_col] > target_mean][target_col].count()
        below_count = df[df[target_col] < target_mean][target_col].count()

        print(f"\n[WEEK4] {target_col} mean: {target_mean:.4f}")
        print(f"[WEEK4] Mean üstü count: {above_count}")
        print(f"[WEEK4] Mean altı count: {below_count}")

        # Çarpıklık yorumu:
        # Eğer mean üstü çok daha fazlaysa -> dağılım bir tarafa yığılmış olabilir.
        # Model kurarken bu dağılımı dikkate almak gerekir (ör. stratify, robust yöntemler, vb.)

    # -----------------------------
    # Sensör kolonlarını seçme
    # -----------------------------
    # Week4 örneğinde 2:53 gibi indexle seçilmişti, ama bu her datasette değişir.
    # En güvenlisi: timestamp ve target dışındaki sayısal kolonlar.
    sensor_cols = [c for c in numeric_cols if c not in [target_col] and c != timestamp_col]
    sensor = df[sensor_cols].copy()

    print("\n[WEEK4] Sensor shape:", sensor.shape)

    # -----------------------------
    # Tek bir sensör için boxplot/hist örnek (V1 varsa)
    # -----------------------------
    example_col = sensor_cols[0] if len(sensor_cols) > 0 else None
    if example_col:
        sns.boxplot(x=sensor[example_col])
        plt.title(f"Boxplot - {example_col}")
        plt.show()

        num_summary(sensor, example_col, plot=True, bins=30)

    # Eğer tüm sensörler için hist/boxplot yapmak istersen:
    # Dikkat: çok fazla grafik açabilir (51 sensör gibi).
    # Ben burada “not defteri” diye kodu bırakıyorum ama default kapalı:
    RUN_ALL_SENSOR_PLOTS = False
    if RUN_ALL_SENSOR_PLOTS:
        for col in sensor_cols:
            num_summary(sensor, col, plot=True, bins=30)

    # -----------------------------
    # Target’a göre sensör ortalamaları
    # -----------------------------
    if target_col in df.columns:
        # target değişkeni continuous olabilir, groupby direkt çalışır ama çok unique değer varsa anlamsız olur.
        # Week4 örneğinde groupby('Target') denmişti, ama Target continuous ise bu tablo çok büyür.
        # O yüzden daha mantıklı: Target’ı bin’lere ayırıp (ör quantile) groupby yapmak.
        #
        # Ama “senin not” aynen dursun diye: örnek bir sensör üzerinden gösteriyorum.
        if example_col:
            try:
                print("\n[WEEK4] df.groupby(Target)[V1].mean() benzeri çıktı (direct):")
                print(df.groupby(target_col)[example_col].mean().head(10))
            except Exception as e:
                print("[WEEK4] groupby direct hata:", e)

    # Daha anlamlı yaklaşım: Target’ı quantile bin’leyelim (opsiyon)
    BIN_TARGET = True
    if BIN_TARGET and target_col in df.columns:
        df["_target_bin"] = pd.qcut(df[target_col], q=5, duplicates="drop")
        if example_col:
            print("\n[WEEK4] Target bin’lerine göre sensör ortalaması (daha anlamlı):")
            print(df.groupby("_target_bin")[example_col].mean())

    # -----------------------------
    # Korelasyon matrisi ve yüksek korelasyon drop
    # -----------------------------
    # Week4 notunda “3’lü gruplar halinde korelasyon” gözlemi vardı.
    corr_sensor = sensor.corr()

    sns.set(rc={"figure.figsize": (12, 12)})
    sns.heatmap(corr_sensor, cmap="RdBu")
    plt.title("Sensor Correlation Heatmap (Before Drop)")
    plt.show()

    drop_list, corr_abs = compute_high_corr_columns(sensor, corr_threshold=corr_threshold)
    print(f"\n[WEEK4] Corr threshold: {corr_threshold}")
    print("[WEEK4] Drop list (yüksek korelasyonlu kolonlar):")
    print(drop_list)

    # Drop uygula
    sensor_reduced = sensor.drop(columns=drop_list, errors="ignore")
    print("\n[WEEK4] Sensor reduced shape:", sensor_reduced.shape)

    # Reduced correlation heatmap
    corr_sensor_reduced = sensor_reduced.corr()
    sns.set(rc={"figure.figsize": (12, 12)})
    sns.heatmap(corr_sensor_reduced, cmap="RdBu")
    plt.title("Sensor Correlation Heatmap (After Drop)")
    plt.show()

    # -----------------------------
    # StandardScaler ile scaling
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sensor_reduced)
    X_scaled_df = pd.DataFrame(X_scaled, columns=sensor_reduced.columns)

    print("\n[WEEK4] X_scaled head():")
    print(X_scaled_df.head())

    # -----------------------------
    # KMeans - optimum k (elbow)
    # -----------------------------
    # Yellowbrick varsa elbow görselleştiriyoruz, yoksa manuel inertia ile bakabiliriz.
    if YELLOWBRICK_AVAILABLE:
        kmeans = KMeans(random_state=random_state)
        elbow = KElbowVisualizer(kmeans, k=(2, 20))
        elbow.fit(X_scaled_df)
        elbow.show()
        best_k = elbow.elbow_value_
        print("\n[WEEK4] Elbow best_k:", best_k)
    else:
        # Yellowbrick yoksa: inertia listesi çıkarıp kendin bakarsın
        inertias = []
        ks = range(2, 21)
        for k in ks:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            km.fit(X_scaled_df)
            inertias.append(km.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(list(ks), inertias, marker="o")
        plt.title("Elbow (manual) - Inertia vs K")
        plt.xlabel("K")
        plt.ylabel("Inertia")
        plt.show()

        # Not: Burada en iyi k’yı otomatik seçmek zor, görselden seçmek daha sağlıklı.
        # Ben default olarak 8 veriyorum, ama sen grafiğe göre ayarla.
        best_k = 8
        print("\n[WEEK4] Yellowbrick yok -> best_k default 8 seçildi. (Grafiğe bakıp değiştir.)")

    # -----------------------------
    # KMeans fit + cluster label ekleme
    # -----------------------------
    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    km_final.fit(X_scaled_df)

    clusters = km_final.labels_
    sensor_clustered = sensor_reduced.copy()
    sensor_clustered["cluster"] = clusters + 1  # cluster id 1’den başlasın diye +1

    print("\n[WEEK4] Cluster counts:")
    print(sensor_clustered["cluster"].value_counts().sort_index())

    # Cluster özet (count/mean/median)
    print("\n[WEEK4] Cluster summary:")
    print(sensor_clustered.groupby("cluster").agg(["count", "mean", "median"]))

    # Cluster CSV export
    clusters_csv_path = os.path.join(out_dir, "clusters.csv")
    sensor_clustered.to_csv(clusters_csv_path, index=False)
    print("\n[WEEK4] clusters.csv kaydedildi:", clusters_csv_path)

    # -----------------------------
    # PCA - explained variance
    # -----------------------------
    # PCA’yı scaled data üzerinde yapmak daha doğru.
    pca = PCA()
    pca_fit = pca.fit_transform(X_scaled_df)

    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    print("\n[WEEK4] PCA explained_variance_ratio_ (ilk 10):")
    print(explained[:10])

    print("\n[WEEK4] PCA cumulative explained variance (ilk 10):")
    print(cum_explained[:10])

    plt.figure(figsize=(8, 4))
    plt.plot(cum_explained)
    plt.title("PCA - Cumulative Explained Variance")
    plt.xlabel("Component count")
    plt.ylabel("Cumulative variance ratio")
    plt.show()

    # “%85 varyansı açıklayan component sayısı” notu:
    n_components_85 = int(np.argmax(cum_explained >= 0.85) + 1)
    print("\n[WEEK4] %85 varyans için gereken bileşen sayısı:", n_components_85)

    # İstersen PCA’yı bu sayıyla tekrar fit edebilirsin:
    pca_85 = PCA(n_components=n_components_85)
    X_pca = pca_85.fit_transform(X_scaled_df)
    X_pca_df = pd.DataFrame(X_pca, columns=[f"PCA{i+1}" for i in range(n_components_85)])
    print("\n[WEEK4] PCA dataframe head():")
    print(X_pca_df.head())

    # -----------------------------
    # Regression (LinearRegression) - örnek
    # -----------------------------
    # Burada iki yaklaşım gösterebilirsin:
    # 1) Direkt sensör_reduced ile regression
    # 2) PCA output ile regression (boyut indirgeme sonrası)
    #
    # Not: Target classification değil regression ise LinearRegression mantıklı.
    # Target binary ise LogisticRegression vs daha doğru olur.
    #
    # Ben burada “Target varsa” diye kontrol koyuyorum.

    if target_col in df.columns:
        y = df[target_col].values

        # (1) Direkt regression (scaled features)
        reg_model_direct = LinearRegression()
        reg_model_direct.fit(X_scaled_df, y)

        print("\n[WEEK4] LinearRegression (direct) intercept:")
        print(reg_model_direct.intercept_)

        print("\n[WEEK4] LinearRegression (direct) coef (ilk 10):")
        print(reg_model_direct.coef_[:10])

        # (2) PCA ile regression
        reg_model_pca = LinearRegression()
        reg_model_pca.fit(X_pca_df, y)

        print("\n[WEEK4] LinearRegression (PCA) intercept:")
        print(reg_model_pca.intercept_)

        print("\n[WEEK4] LinearRegression (PCA) coef (tümü):")
        print(reg_model_pca.coef_)

        # Tahminleri export (PCA üzerinden örnek)
        y_pred_pca = reg_model_pca.predict(X_pca_df)
        pred_path = os.path.join(out_dir, "y_pred_pca.csv")
        pd.DataFrame({"y_pred_pca": y_pred_pca}).to_csv(pred_path, index=False)
        print("\n[WEEK4] y_pred_pca.csv kaydedildi:", pred_path)

    # -----------------------------
    # Kaydetme (scaler + kmeans + pca)
    # -----------------------------
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(km_final, os.path.join(out_dir, "kmeans.pkl"))
    joblib.dump(pca_85, os.path.join(out_dir, "pca.pkl"))

    print("\n[WEEK4] Kaydedildi:", os.path.join(out_dir, "scaler.pkl"))
    print("[WEEK4] Kaydedildi:", os.path.join(out_dir, "kmeans.pkl"))
    print("[WEEK4] Kaydedildi:", os.path.join(out_dir, "pca.pkl"))

    return {
        "df": df,
        "sensor_reduced": sensor_reduced,
        "sensor_clustered": sensor_clustered,
        "scaler": scaler,
        "kmeans": km_final,
        "pca": pca_85
    }

# ============================================================
# 5) MAIN (dosya direkt çalıştırılırsa burası çalışsın diye)
# ============================================================

if __name__ == "__main__":
    # Burayı “ben bu dosyayı çalıştırdım, pipeline’lar sırayla aksın” diye koydum.
    # İstersen birini kapatıp diğerini çalıştırabilirsin.

    # -----------------------------
    # A) HuggingFace metin pipeline
    # -----------------------------
    RUN_HF = True
    if RUN_HF:
        hf_results = run_hf_offensive_language_pipeline(
            hf_dataset_name="Toygar/turkish-offensive-language-detection",
            text_col="text",
            label_col="label",
            test_size=0.20,
            random_state=42,
            model_out_dir="outputs_hf_offensive"
        )

    # -----------------------------
    # B) Week4 sensör pipeline
    # -----------------------------
    RUN_WEEK4 = False
    # Çünkü sende csv_path farklı olacak; ben placeholder bıraktım.
    # Açmak için RUN_WEEK4=True yap, csv_path’i doğru ver.
    if RUN_WEEK4:
        week4_results = run_week4_sensor_pipeline(
            csv_path="/Users/nisacayir/ACM476/data_ready_for_cs.csv",  # <-- BUNU DEĞİŞTİR
            target_col="Target",
            timestamp_col="Timestamp",
            corr_threshold=0.90,
            random_state=17,
            out_dir="outputs_week4_sensor"
        )

    # Not:
    # Eğer iki pipeline’ı aynı anda açarsan çok grafik çıkacak.
    # Normalde önce birini bitirip sonra diğerine geçmek daha rahat.
