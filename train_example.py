import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from category_encoders import TargetEncoder
from catboost import CatBoostRegressor

# Criar um DataFrame de exemplo
data = {
    'cat_feature1': ['A', 'B', 'A', 'B', 'C'],
    'cat_feature2': ['X', 'Y', 'Y', 'X', 'X'],
    'num_feature1': [1.0, 2.5, 3.5, 4.0, 5.0],
    'num_feature2': [10, 20, 30, 40, 50],
    'prob_feature1': [0.1, 0.4, 0.6, 0.8, 0.2],
    'prob_feature2': [0.2, 0.3, 0.5, 0.7, 0.9],
    'target': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

# Separar features e alvo
X = df.drop('target', axis=1)
y = df['target']

# Separar tipos de features
categorical_features = ['cat_feature1', 'cat_feature2']
numerical_features = ['num_feature1', 'num_feature2']
probability_features = ['prob_feature1', 'prob_feature2']

# Pipeline para features categóricas
cat_transformer = Pipeline(steps=[
    ('target_encoder', TargetEncoder(cols=categorical_features)),
])

# Pipeline para features numéricas
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Criar um ColumnTransformer para as features categóricas e numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', cat_transformer, categorical_features),
        ('numerical', num_transformer, numerical_features),
        ('probability', 'passthrough', probability_features)  # Manter as features de probabilidade como estão
    ]
)

# Modelos base para o Stacking
base_models = [
    ('catboost', CatBoostRegressor(cat_features=categorical_features, verbose=0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
]

# Pipeline de Stacking
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)  # Modelo meta
)

# Criar a pipeline final
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking_model)
])

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
final_pipeline.fit(X_train, y_train)

# Fazer previsões
predictions = final_pipeline.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.4f}')
