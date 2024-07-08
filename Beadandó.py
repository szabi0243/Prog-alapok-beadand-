import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

fajl = pd.read_csv('This PC/Desktop/Egyetem/stadat-nep0064-22.2.1.1-hu.csv', sep=';', header=1, encoding='utf-8',
                   skiprows=0)
df = pd.read_csv(delimiter=';')
print(df.head())
plt.figure(figsize=(12, 8))


def fajlinfo():
    print(fajl.head())
    print(fajl.describe())
    print(fajl.tail())
    print(fajl.info())
    print(fajl.corr())


def HaviMeresek():
    global havi_szam
    havi_szam = len(fajl) - 1
    print(f"(str{(havi_szam)} havi adati részletességet tartalmazza.")


plt.subplot(2, 2, 1)
sns.lineplot(x='Év', y='A házasságkötések száma', data=df, marker='o', label='Házasságkötések')
sns.lineplot(x='Év', y='Az élveszületések száma', data=df, marker='o', label='Élveszületések')
plt.title('Házasságkötések és élveszületések alakulása')

plt.subplot(2, 2, 2)
sns.lineplot(x='Év', y='A halálozások száma', data=df, marker='o', color='red', label='Halálozások')
plt.title('Halálozások alakulása')

plt.subplot(2, 2, 3)
sns.lineplot(x='Év', y='A természetes szaporodás, fogyás száma', data=df, marker='o', color='green',
             label='Természetes szaporodás')
plt.title('Természetes szaporodás alakulása')

plt.subplot(2, 2, 4)
x = df['Évkezdettől akkumulált'].values.reshape(-1, 1)
y = df['A házasságkötések száma'].values
model = LinearRegression().fit(x, y)
plt.scatter(x, y, color='blue', label='Házasságkötések')
plt.plot(x, model.predict(x), color='red', linewidth=2, label='Lineáris regresszió')
plt.title('Lineáris regresszió az évkezdettől akkkumulált házasságkötések alapján')

plt.tight_layout()
plt.show()
