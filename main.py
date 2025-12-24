from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib
from sklearn.preprocessing import StandardScaler

# dataset'i HF'den yükleme
dataset = load_dataset("Toygar/turkish-offensive-language-detection")

# train verisini pandas data frame'e dönüştürme
df = dataset["train"].to_pandas()

# İlk birkaç satırı göster
print(df.head())
print(f"Toplam satır sayısı: {len(df)}")

#cmd alt e

# Genel bilgi
print(df.info())


# Eksik veri kontrolü
print(df.isnull().sum())

# Kaç adet 0  ve  1 var?
print(df['label'].value_counts())

# Yüzdelik oran
print(df['label'].value_counts(normalize=True) * 100)

# Tweet uzunluklarını hesapla
df['tweet_length'] = df['text'].apply(len)

# Ortalama, min, max uzunluk
print(df['tweet_length'].describe())

# Histogram grafiği
plt.hist(df['tweet_length'], bins=50)
plt.title("Tweet Length Distribution")
plt.xlabel("Character count")
plt.ylabel("Frequency")
plt.show()

# Kelime sayısı
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
print(df['word_count'].describe())

# En uzun ve en kısa tweet
print(df.loc[df['word_count'].idxmax()])
print(df.loc[df['word_count'].idxmin()])


# Saldırgan vs temiz tweet uzunluk karşılaştırması
print(df.groupby('label')['tweet_length'].mean())


#df.shape sık sık kullanırız : shape tüm şeklini getiriyor yani x row y col gibi gibi

print(df.describe(include='object'))


df['label'] = df['label'].astype(int)
mean_label = df['label'].mean()
filtered_df = df.loc[df['label'] >= mean_label]
print(filtered_df.head())

#Bu, label değeri ortalamanın (mean) üzerinde olan satırları getirir.
#Ama senin dataset’inde label 0 ve 1 olduğu için ortalaması 0.47 civarındadır
#dolayısıyla bu, label == 1 olan satırları getirir (yani offensive olanları)

sns.boxplot(x=df['word_count'], color='orange')
plt.title("Kelime Sayısı Dağılımı")
plt.show()

def num_summary(dataframe, numerical_col, plot=True):
    print("-----------", numerical_col, "-----------")
    print(dataframe[numerical_col].describe())   # betimleyici istatistiklerin bulunduğu kısım
    print("\nQuantiles:")
    print(dataframe[numerical_col].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(f"Distribution of {numerical_col}")
        plt.show()

def target_summary_with_num(dataframe, target, numerical_col):
    print("-----------", numerical_col, "-----------")
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print("-----------", numerical_col, "-----------")
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

# label hariç tüm sayısal sütunları seçelim
num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'label']

# hepsine uygularsak
for col in num_cols:
    target_summary_with_num(df, "label", col)

corr = df.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Korelasyon Isı Haritası")
plt.show()
# iki aynı variable aynı olunca birbirinin pozitif +1 korelasyonudur
# korelasyon matrisi birbirinin simetrisi ve o şekilde düşünülmeli


corr_matrix = df.corr(numeric_only=True).abs()  # mutlak değerli korelasyon matrisi
# üst üçgeni alalım
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 0.90 üzerindeki korelasyonları bul
high_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.90)]

print("Silinecek yüksek korelasyonlu değişkenler:")
print(high_corr)

df_reduced = df.drop(columns=high_corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Orijinal Korelasyon Matrisi +++")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df_reduced.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Korelasyon Azaltılmış Reduced Frame")
plt.show()


# X senin bağımsız değişkenlerin olsun (örneğin sayısal kolonlar)
X = df[['tweet_length', 'word_count']]   # örnek: sayısal değişkenler

# StandardScaler ile veriyi standardize et (ortalama=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Yeniden DataFrame’e çevir
X_scaled_v1 = pd.DataFrame(X_scaled, columns=X.columns)

print(X_scaled_v1.head())


