import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')

if 'Time' in data.columns:
    data.drop(['Time'], axis=1, inplace=True)

sns.set(style="whitegrid")

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data.drop(['Date'], axis=1, inplace=True)

def plot_distribution(data, columns):
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.histplot(data[column].dropna(), kde=True, bins=30)
        plt.title(f'Распределение {column}')
    plt.show()

plot_distribution(data, ['Aboard', 'Fatalities', 'Ground'])

correlation_matrix = data[['Aboard', 'Fatalities', 'Ground']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Корреляционная матрица')
plt.show()

descriptive_stats = data[['Aboard', 'Fatalities', 'Ground']].describe()
print("Основные статистические характеристики:")
print(descriptive_stats)

missing_values = data.isnull().mean() * 100
print("\nПроцент пропущенных значений в каждом столбце:")
print(missing_values[missing_values > 0])

data['Aboard'].fillna(data['Aboard'].median(), inplace=True)
data['Fatalities'].fillna(data['Fatalities'].median(), inplace=True)
data['Ground'].fillna(data['Ground'].median(), inplace=True)
data.dropna(subset=['Location'], inplace=True)

categorical_columns = data.select_dtypes(include=['object']).columns

data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

quantitative_columns = ['Aboard', 'Fatalities', 'Ground', 'Year']
scaler = MinMaxScaler()
data_encoded[quantitative_columns] = scaler.fit_transform(data_encoded[quantitative_columns])

data_encoded['High_Fatalities'] = (data_encoded['Fatalities'] > data_encoded['Fatalities'].median()).astype(int)

X = data_encoded.drop(['Aboard', 'Fatalities', 'High_Fatalities'], axis=1, errors='ignore')
y = data_encoded['High_Fatalities']

print("\n Число объектов:", data.shape[0])
print(" Количество классов в целевой переменной:", y.nunique())
print("Пропущенные значения (в процентах):")
print(missing_values[missing_values > 0])

class_distribution = y.value_counts(normalize=True) * 100
print("\nСоотношение классов (в процентах):\n", class_distribution)

Q1 = data[['Aboard', 'Fatalities', 'Ground']].quantile(0.25)
Q3 = data[['Aboard', 'Fatalities', 'Ground']].quantile(0.75)
IQR = Q3 - Q1
outliers = ((data[['Aboard', 'Fatalities', 'Ground']] < (Q1 - 1.5 * IQR)) | (data[['Aboard', 'Fatalities', 'Ground']] > (Q3 + 1.5 * IQR))).sum()
outlier_percentage = (outliers / data.shape[0]) * 100
print("\nПроцент выбросов в количественных признаках:")
print(outlier_percentage)

text_columns = data.select_dtypes(include=['object']).columns
print("\nНаличие текстовых признаков:", list(text_columns) if len(text_columns) > 0 else "Отсутствуют")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("\nТочность модели k-ближайших соседей:", accuracy_score(y_test, y_pred))
print("\nОтчет классификации:\n", classification_report(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
plt.title('Матрица ошибок для k-ближайших соседей')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.show()
