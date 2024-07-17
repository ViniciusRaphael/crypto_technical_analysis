import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Carregar dataset
iris = load_iris()
X, y = iris.data, iris.target

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Salvar o modelo treinado
joblib.dump(clf, 'random_forest_model.joblib')

# Carregar o modelo treinado
clf_loaded = joblib.load('random_forest_model.joblib')

# Fazer previsões
predictions = clf_loaded.predict(X_test)
print(predictions)
