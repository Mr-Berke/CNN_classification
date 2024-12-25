import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from category_encoders.one_hot import OneHotEncoder

def function():
    df=pd.read_csv("C:\\Users\\Berke\\Desktop\\ysa_final_odev\\6_class_star.csv")
    # veri temizleme ve normalizasyon
    # renkler küçük harfe çevrildi , boşluklar kaldırıldı. ve bazı renk grupları bileştirilerek sadeleştirme yapıldı.
    df["Star color"]=df["Star color"].str.lower().str.replace(" ","").replace("-","")
    df["Star color"]=df["Star color"].str.replace("yellowishwhite", "yellowwhite")\
                                  .str.replace("whiteyellow", "yellowwhite")\
                                  .str.replace("whitish", "white")

    # log dönüşümü ve normalleştirme 
    # değerler arası fark çok fazla olduğu için log alınarak sabit değere çekildi
    df["Temperature (K)"]=np.log(df["Temperature (K)"])
    df["Luminosity(L/Lo)"]=np.log(df["Luminosity(L/Lo)"])
    df["Radius(R/Ro)"]=np.log(df["Radius(R/Ro)"])

    # değerler 0-1 arasına normalize edilir
    scaler=MinMaxScaler()
    df["Temperature (K)"]=scaler.fit_transform(np.expand_dims(df["Temperature (K)"],axis=1))
    df["Luminosity(L/Lo)"]=scaler.fit_transform(np.expand_dims(df["Luminosity(L/Lo)"],axis=1))
    df["Radius(R/Ro)"]=scaler.fit_transform(np.expand_dims(df["Radius(R/Ro)"],axis=1))
    df["Absolute magnitude(Mv)"]=scaler.fit_transform(np.expand_dims(df["Absolute magnitude(Mv)"],axis=1))

    # spektral sınıfı sayısal değere dönüştürme
    df["Spectral Class"] = df["Spectral Class"].map({"M": 0, "K": 1, "G": 2, "F": 3, "A": 4, "B": 5, "O": 6})

    # one-hot encoding uygulması
    # yıldızlar sütununun renklerini her birini farklı sütunlaa bölme ve 1 0 şeklinde belrtme
    one_hot=OneHotEncoder(cols=["Star color"],use_cat_names=True)
    df=one_hot.fit_transform(df)

    # özellik(x) ve hedef değişken ayırma(y)
    x=df.drop(columns=["Star type"])
    y=df["Star type"]
    return x,y
function()