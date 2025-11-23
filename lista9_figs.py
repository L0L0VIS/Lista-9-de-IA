import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ======================================================
# 1) Carregar e visualizar
# ======================================================
df = pd.read_csv("creditcard.csv")

print("\n===== VISUALIZAÇÃO INICIAL =====")
print(df.head())
print(df.info())
print(df.describe())

# Salvando histograma geral
plt.figure(figsize=(12,6))
df['Amount'].hist(bins=50)
plt.title("Distribuição do atributo Amount")
plt.savefig("amount_hist.png")
plt.close()

# ======================================================
# 2) Missing values
# ======================================================
print("\n===== MISSING VALUES =====")
print(df.isna().sum())

# ======================================================
# 3) Redundância e inconsistência
# ======================================================
print("\n===== DUPLICADOS =====")
print("Duplicados:", df.duplicated().sum())

df_nodup = df.drop_duplicates()

# ======================================================
# 4) Outliers — Amount e Time
# ======================================================
plt.figure(figsize=(10,5))
sns.boxplot(x=df_nodup['Amount'])
plt.title("Boxplot Amount")
plt.savefig("boxplot_amount.png")
plt.close()

plt.figure(figsize=(10,5))
sns.boxplot(x=df_nodup['Time'])
plt.title("Boxplot Time")
plt.savefig("boxplot_time.png")
plt.close()

# ======================================================
# 5) Normalização
# ======================================================
scaler = StandardScaler()
df_nodup['Amount_scaled'] = scaler.fit_transform(df_nodup[['Amount']])
df_nodup['Time_scaled']   = scaler.fit_transform(df_nodup[['Time']])

# ======================================================
# 6) Correlação
# ======================================================
plt.figure(figsize=(15,10))
sns.heatmap(df_nodup.corr(), cmap="coolwarm", center=0)
plt.title("Mapa de Correlação")
plt.savefig("correlation_heatmap.png")
plt.close()

# ======================================================
# 7) Codificação de variáveis
# (não há variáveis categóricas — base já está codificada)
# ======================================================
print("\nNenhuma variável categórica para codificar.")

# ======================================================
# 8) Balanceamento da classe
# ======================================================
X = df_nodup.drop("Class", axis=1)
y = df_nodup["Class"]

print("Distribuição antes do SMOTE:")
print(y.value_counts())

sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

print("Distribuição após o SMOTE:")
print(pd.Series(y_res).value_counts())

# ======================================================
# 9) Train-test split estratificado
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
)

print("\nShapes finais:")
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

print("\nPRÉ-PROCESSAMENTO FINALIZADO COM SUCESSO!")
