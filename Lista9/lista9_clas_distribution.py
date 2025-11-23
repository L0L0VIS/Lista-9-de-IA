import pandas as pd
import matplotlib.pyplot as plt

# Carrega o dataset (ajuste o caminho se necessário)
df = pd.read_csv("creditcard.csv")

# Contagem das classes
class_counts = df['Class'].value_counts().sort_index()

# Plot
plt.figure(figsize=(6,4))
plt.bar(class_counts.index, class_counts.values)
plt.xticks([0,1], ['Classe 0 (Legítima)', 'Classe 1 (Fraude)'])
plt.ylabel('Número de Transações')
plt.title('Distribuição das Classes')

# Mostra proporção no topo das barras
for i, v in enumerate(class_counts.values):
    plt.text(i, v + max(class_counts.values)*0.01, str(v), ha='center', fontsize=10)

# Salva a figura
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()