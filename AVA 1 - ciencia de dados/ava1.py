import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt


url = './asset/waterQuality1.csv'  # Substitua pelo caminho do seu arquivo CSV
try:
    df = pd.read_csv(url)
    print("Arquivo CSV carregado com sucesso!")
except pd.errors.ParserError as e:
    print(f"Erro ao carregar CSV: {e}")
    exit()

# 1 - Análise inicial
print("Primeiras linhas do DataFrame:")
print(df.head())
print("\nÚltimas linhas do DataFrame:")
print(df.tail())
print("\nTipos de dados das colunas:")
print(df.dtypes)

# 2- Identificação e remoção de registros com problemas nas colunas 'ammonia' e 'is_safe'
print("\nValores únicos em 'ammonia':")
print(df['ammonia'].value_counts())
print("\nValores únicos em 'is_safe':")
print(df['is_safe'].value_counts())

# 3 - Converter colunas 'ammonia' e 'is_safe' para numéricas
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')

# 4 - Remover registros com NaN nas colunas 'ammonia' e 'is_safe'
df = df.dropna(subset=['ammonia', 'is_safe'])
print("\nDados após remoção de NaNs:")
print(df.dtypes)

# 5 - Balanceamento dos dados usando SMOTE
X = df.drop('is_safe', axis=1)
y = df['is_safe']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("\nDistribuição da variável alvo após SMOTE:")
print(pd.Series(y_res).value_counts(normalize=True))

# 6 - Análise exploratória de dados
sns.histplot(df['ammonia'], kde=True)
plt.title('Distribuição de Ammonia')
plt.show()

sns.histplot(df['copper'], kde=True)
plt.title('Distribuição de Copper')
plt.show()

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# 7 - Separação dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# 8 - Aplicação dos classificadores e avaliação de performance
# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("\nGaussian Naive Bayes:")
print(classification_report(y_test, y_pred_gnb))

# K Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("\nK Nearest Neighbours:")
print(classification_report(y_test, y_pred_knn))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("\nDecision Tree:")
print(classification_report(y_test, y_pred_dt))
