#ENVIRONMENT SET UP

#which env am I in conda
#conda env list #opt + shift + e (satır bazlı çalıştırmak için)  alt+shift+e (windows için)

#Which packages are available in my env
#conda list

#I am exporting the package environment needed for the New project from the base env to a file
#conda env export > environment.yaml

#create a new env for the new project
#conda create -n acm4762025_env

#activating the New project virtual env
#conda activate acm4762025_env

#I accept base as origin and use the .yaml file for acm4762025_env
#conda env create -f environment.yaml

#listing which packages are available
#conda list

#updating all packages
#conda upgrade -all (Patlayabilir çünkü bazı paketler birbirleriyle uyumsuz olabilir)

#DATA ANALYSIS
#Starting data exploration

import numpy as np
import pandas as pd
from fontTools.unicodedata import block

#reading the sheet that contains raw data from the CSV file

df = pd.read_csv("/Users/gizemmervedemir/ACM476/data_ready_for_cs.csv")

#displaying the first 5 rows for verification
df.head

#step 1: performing descriptive statistics analyses

#Making a separate definition for numerical variables
num_var = [col for col in df.columns if df[col].dtype != '0']
num_var

#listing the basic descriptive functions
desc_agg = ['sum', 'mean' , 'std' , 'var' , 'min' , 'max']
desc_agg

#applying these functions to numerical values
desc_agg_dict = { col : desc_agg for col in df}
desc_agg_dict

desc_summ = df[num_var].agg(desc_agg_dict)

#printing desc_summ to examine each variable's sum, mean, standard deviation, min and max values
print(desc_summ)

#I want to convert it to a numpy array
df_desc_na = np.array(desc_summ) #tek boyutlu matrise çevirdik her sütun var o zamana ait her şey var tek boyuta indirdik numpy fonksiyonları da arrayleştirmeden rahat çalışmıyor
df_desc_na

#to use df as a numpy array; for vector operations, etc.
df_na = np.array(df)
df_na

#Continuing Overview

import seaborn as sns

df.shape

df.info() #diğerleri parantezsiz bu parantezli
df.columns
#tüm data non-null MISSING DATA YOK, tüm veri float numeric, sensör dışı 2 data var: target, tiemstamp, 50 sensor data

#missing value için ekstra kontrol yapıyorum FALSE geliyor
df.isnull().values.any() #false

#her bir depşşkene ait descriptive analytics değerleri bir tabloya yeniden yazdırıyorum
desc_summv2 = df.describe().T #transpose column-> row

#Target'ı inceleyelim

df[df.Target > df.Target.mean()].Target.count() #27485
df[df.Target < df.Target.mean()].Target.count() #21472

#data çarpık
#lab deney sonuçları ortalamanın sağında yayılımlı (sola skewed)
#tahminlerimizde bu çarpıklık dikkate alınmalı

df.loc[df.Target > df.Target.mean(), 'V1'].head()

#bütün sensör verilerinde Target değişkenine paralel pozitif çarpık sensörleri inceliyorum
sensor = df.iloc[:, 2:53] #index bazlı çalışıyor ikinci kolondan 53.kolona kadar bana getir demek, index 0 dan başlar
sensor #sen kendi veri setine göre isimlendirirsin

sensor.columns

#değişkenlerin görseller ile incelenmesi
from matplotlib import pyplot as plt

#değişkenlerin grafiklerini çıkarıyorum
sns.boxplot(x = sensor['V1'])
plt.show()

def num_summary(data, numerical_col, plot=True):
    quantiles=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,0.80,0.90, 0.95, 0.99] #aykırı değer yok
    print(data[numerical_col].describe(quantiles).T)

    if plot:
        data[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(sensor,'V1', plot=True)

#tüm değişkenler için bir kod ile grafikler üretiyorum
for col in sensor:
    num_summary(sensor,col,plot=True)

#tüm veri seti sayısal olduğu için, değişken yakalamak veya genelleme sürecini atlıyorum

#bağımlı dğeişkenin bağımsız değişkenler üzerinden analiz ediyorum

df.groupby('Target')['V1'].mean()

#target -> tahmin etmeye çalıştığın şey

#tüm sayısal değerler için bu fonksiyonu itere etmek istiyorum

def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: 'mean'}), end="\n\n\n")

target_summary_with_num(df,'Target','V1')

#fonksiyon 51 sensor için de çalışsın
for col in sensor:
    target_summary_with_num(df,'Target',col)

#peki sensorler arasında korelasyon nasıl, model optimizasyonu için korelasyona göre ayıklanması değerlendiriyorum

#tüm korelasyonları çıkarıyorum
corr = sensor.corr()

#korelasyon ısı haritası çıkartmak istiyorum
sns.set(rc={'figure.figsize' : (12, 12)})
sns.heatmap(corr, cmap='RdBu')
plt.show()

#sensorlerlerin 3'lü gruplar halinde korelasyonlarının ortak olduğunu fark ediyorum
#ayrıca korelasyonun yğksek olduğu değişkenleri ve ayıklamak
cor_matrix = df.corr().abs

upper_triangle_matrix = cor_matrix_where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]

#veri setlerimden (hem) yoğun korele değişkenleri siliyorum
df = df.drop(drop_list, axis=1)
sensor = sensor.drop(drop_list, axis=1)

sensor.shape
#yola 24 sensor ile devam edeceğim, heatmapte de aynı şekilde 3'erli gruplarda görülüyor sensör değerleri

#Model development

#Modeling
#Prediction
#Evaluation
#Hyperparameter optimization
#Finalization

#Roadmap oalrak bu şekilde ilerliyor olacağız

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
#pd.set_option('display.max_columns', None) #veriyi küçülttüğüm için kaldırıyorum

Y = df.iloc[: , 1:2] #bağımlı
X = df.drop(['Target', 'Timestamp'], axis=1) #bağımsız

#bağımsız değişkenleri standardize ediyoruz

X_scaled = StandardScaler().fit_transform(X)
X_scaled_v1 = pd.DataFrame(X_scaled, columns=X.columns)

#öznitelikleri(feature) kümeleme yapmak istiyorum, --clustering modelling
#sonra her bir satırın yanına sınıfını ekleyip, regresyon modeli oluşturacağız
#PCA da denenebilir

import matplotlib.pyplot as plt
import yellowbrick as yb
from sklearn.cluster import KMeans #dbscan ve k = medo'ds de araştıralabilir
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#optimum küme sayısını belirliyorum

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(X_scaled_v1)
elbow.show()

elbow.elbow_value_

#bu cluster sayısı ile k-means çalıştırıyoruz

kmeans = Kmeans(n_clusters= elbow.elbow_value_, random_state=17).fit(X_scaled_v1) #17-20 arası olsa iyi olur
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_ #bunlar alt çizgili böyle kullanmalısın
kmeans.inertia_ #bütün yaygınlığı karmaşıklığı ölçüyor

#label'ları cluster olarak tanımladım
clusters = kmeans.labels_

#0'dan başlamamalaırnı istiyorum- cluster id'ler 1-8 arası olacak
sensor['cluster'] = clusters + 1

sensor.head()
sensor.columns

# ---------------------------------------------------------
# Buradan sonra B ayrı cluster modeli train edip
# yeni değerlerde en iyi çalışan modeli kabul etmeyi tercih ederim.
# ---------------------------------------------------------

sensor.groupby('cluster').agg(['count', 'mean', 'median'])


# ---------------------------------------------------------
# Train veri setinin kümelerini bir CSV dosyasına yazalım
# ---------------------------------------------------------

sensor.to_csv('clusters.csv')

# ---------------------------------------------------------
# Cluster'ları incelediğimde temel bileşenler analizi (PCA) yapmak istiyorum
# ---------------------------------------------------------

pca = PCA() #birbirinden bağımsız aynı sonuca çıkmayan ifade etmeyen değişkenlerin olduğu yerde pca çok da gerekli değil
pca_fit = pca.fit_transform(X_scaled_v1)

# Açıklanan varyans oranı
pca.explained_variance_ratio_

np.cumsum(pca.explained_variance_ratio_) #cumulative sonuç eğer 1 ise demek ki etken değil benim için target için analitik geometrideki gibi

#ilk 8 değişkenle mevcut değişkenliğin %85ini açıklayabiliyorum

# Kümülatif varyans grafiği
pca = PCA().fit(X_scaled_v1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Bileşen Sayısı')
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

#PCA Modeli
pca=PCA(n_components=8)
pca_fit = pca.fit_transform(X_scaled_v1)
pca.explained_variance_ratio_

np.cumsum(pca.explained_variance_ratio_)

####İLETİLEN TEST VERİSİNE UYGULUYORUM

df_test_PCA = pd.read_excel(".xlsx dosyası")

X_test_targetsiz_PCA = df_test_PCA.drop(['Timestamp'], axis=1)

X_test_targetsiz_PCA = X_test_targetsiz_PCA.dropna()

pca_fit = pca.fit_transform(X_test_targetsiz_PCA)
pca.explained_variance_ratio_

np.cumsum(pca.explained_variance_ratio_)

X_test_targetsiz_PCA.head

RMSE_PCA=np.cumsum(pca.explained_variance_ratio_)

#8 bileşenin 2 tanesi veya 3 tanesi modeli açıklıyor görünüyor

pd.DataFrame(pca_fit, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8'])

final_df_targetsiz = pd.concat([pd.DataFrame(pca_fit, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8']), Y], axis=)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

y = df['Target']
x = df.drop(['Timestamp', 'Target'], axis=1)

df_train = pd.read_excel('xlsx file', sheet_name='Train')
df_test = pd.read_excel('xlsx file', sheet_name='Test')

y_train = df_train['Target']
x_train = df_train.drop(['Timestamp', 'Target'], axis=1)

x_train.isnull().values.any()

y_test = df_test['Target']
X_test = df_test.drop([
    'Timestamp', 'V2', 'V3', 'V8', 'V9',
    'V11', 'V12', 'V17', 'V18', 'V23', 'V25', 'V26', 'V27',
    'V29', 'V30', 'V32', 'V33', 'V35', 'V36', 'V38', 'V39',
    'V41', 'V42', 'V44', 'V45', 'V50', 'V51'
], axis=1)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

reg_model_df = LinearRegression().fit(X, y)
reg_model = LinearRegression().fit(X_train, y_train)

# bias, sabit
reg_model_df.intercept_
reg_model.intercept_

# coefficient, weight
reg_model_df.coef_
reg_model.coef_

X_test_targetsiz_PCA = X_test_targetsiz_PCA.dropna()

y_pred_test_targetsiz_PCA = reg_model.predict(X_test_targetsiz_PCA)

#documentation

y_pred_test_targetsiz_PCA = pd.DataFrame(y_pred_test_targetsiz_PCA)

y_pred_test_targetsiz_PCA['y_pred_test_targetsiz_PCA'] = y_pred_test_targetsiz_PCA

y_pred_test_targetsiz_PCA.to_csv('y_pred_test_targetsiz_PCA.csv')

#####İLETİLEN TEST VERİSİNE UYGULADIM (SON)




