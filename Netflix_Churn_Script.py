#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

from datetime import datetime
import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

import shap

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

import joblib
import numpy as np

# 1. Carga los datos
users_df = pd.read_csv(r"PATH")
movies_df = pd.read_csv(r"PATH")
watch_history_df = pd.read_csv(r"PATH")

# 2. Limpia de manera sencilla el DataFrame de 'usuarios'
users_df['age'] = users_df['age'].fillna(users_df['age'].median())
users_df = users_df[(users_df['age'] > 10) & (users_df['age'] < 100)]

# 3. Fusiona en un DataFrame
# Este es el inicio importante del proceso
# (user_id y movie_id son claves comunes)
merged_df = pd.merge(watch_history_df, users_df, on='user_id', how='left')
merged_df = pd.merge(merged_df, movies_df, on='movie_id', how='left')

# El 'merged_df' ya está listo para los siguientes pasos de ingeniería de características y el modelado.
merged_df.head()

#EXPLORACION DE LOS DATOS
# Convierte la columna de fechas a formato de fecha y hora por si no lo está (garantiza el formato que utilices)
merged_df['watch_date'] = pd.to_datetime(merged_df['watch_date'])

# Encontra la fecha más reciente de actividad en todo el dataset
latest_date = merged_df['watch_date'].max()

# Encontra la última fecha de visualización para cada usuario
last_watch_date = merged_df.groupby('user_id')['watch_date'].max().reset_index()
last_watch_date.rename(columns={'watch_date': 'last_active_date'}, inplace=True)

# Calcula los días de inactividad para cada usuario
last_watch_date['days_inactive'] = (latest_date - last_watch_date['last_active_date']).dt.days

# Fusiona los días de inactividad de vuelta al DataFrame principal
merged_df = pd.merge(merged_df, last_watch_date[['user_id', 'days_inactive']], on='user_id', how='left')

# Define la columna 'is_churned' basada en los umbrales de inactividad
# Creamos tres columnas para cada umbral (30, 60, y 90 días) para que se pueda comprobar segun el criterio de cada plataforma
merged_df['is_churned_30'] = (merged_df['days_inactive'] > 30).astype(int)
merged_df['is_churned_60'] = (merged_df['days_inactive'] > 60).astype(int)
merged_df['is_churned_90'] = (merged_df['days_inactive'] > 90).astype(int)

print(merged_df[['user_id', 'days_inactive', 'is_churned_30', 'is_churned_60', 'is_churned_90']].head())


#PROPONER LAS METRICAS CLAVE AQUI
# Agrupa por usuario para calcular métricas de engagement
user_engagement = merged_df.groupby('user_id').agg(
    total_watch_duration_minutes=('watch_duration_minutes', 'sum'),
    avg_watch_duration_per_session=('watch_duration_minutes', 'mean'),
    num_sessions=('session_id', 'count')
).reset_index()

# Calcula la frecuencia de sesiones por semana.
# Primero, calculamos la duración total del usuario en días y luego la convertimos a semanas
user_duration = merged_df.groupby('user_id')['watch_date'].agg(
    min_date='min', max_date='max'
).reset_index()

user_duration['total_days_active'] = (user_duration['max_date'] - user_duration['min_date']).dt.days + 1
user_duration['total_weeks_active'] = user_duration['total_days_active'] / 7

# Une el número de sesiones con las semanas activas para calcular la frecuencia
user_engagement = pd.merge(user_engagement, user_duration[['user_id', 'total_weeks_active']], on='user_id')
user_engagement['sessions_per_week'] = user_engagement['num_sessions'] / user_engagement['total_weeks_active']

# Une estas nuevas características al DataFrame principal para el modelado
merged_df = pd.merge(merged_df, user_engagement[['user_id', 'total_watch_duration_minutes', 
                                              'avg_watch_duration_per_session', 'sessions_per_week']], on='user_id')

print(merged_df[['user_id', 'total_watch_duration_minutes', 'avg_watch_duration_per_session', 'sessions_per_week']].head())

# Crea una lista de todos los géneros para cada sesión.
merged_df['all_genres'] = merged_df.apply(
    lambda row: [row['genre_primary'], row['genre_secondary']], axis=1
)

# Agrupa por usuario para obtener una lista de todos los géneros únicos que han visto.
user_content_exploration = merged_df.groupby('user_id')['all_genres'].sum().apply(
    lambda x: pd.Series(x).explode().unique().tolist()
).reset_index()

# Cuenta el número de géneros únicos vistos.
user_content_exploration['unique_genres_watched'] = user_content_exploration['all_genres'].apply(len)

# También puedemos considerar agregar el número de películas únicas que se han visto, como antes usamos.
user_content_exploration['unique_movies_watched'] = merged_df.groupby('user_id')['movie_id'].nunique()

# Combina de nuevo con el DataFrame principal
merged_df = pd.merge(merged_df, user_content_exploration[['user_id', 'unique_genres_watched', 'unique_movies_watched']], on='user_id')

print(merged_df[['user_id', 'unique_genres_watched', 'unique_movies_watched']].head())

# Calcula la duración total de visualización para cada usuario.
user_watch_duration = merged_df.groupby('user_id')['watch_duration_minutes'].sum().reset_index()
user_watch_duration.rename(columns={'watch_duration_minutes': 'total_watch_time'}, inplace=True)

# Calcula la duración total de visionado de películas para cada usuario (podriamos considerar otro tipo de formato).
movie_watch_duration = merged_df[merged_df['content_type'] == 'Movie'].groupby('user_id')['watch_duration_minutes'].sum().reset_index()
movie_watch_duration.rename(columns={'watch_duration_minutes': 'movie_watch_time'}, inplace=True)

# Combina los dos DataFrames
user_metrics = pd.merge(user_watch_duration, movie_watch_duration, on='user_id', how='left').fillna(0)

# Calcula la proporción de películas vistas (dato util para comparacion)
user_metrics['movie_watch_ratio'] = user_metrics['movie_watch_time'] / user_metrics['total_watch_time']

# Combina esta nueva función en tu DataFrame principal.
merged_df = pd.merge(merged_df, user_metrics[['user_id', 'movie_watch_ratio']], on='user_id', how='left')

print(merged_df[['user_id', 'movie_watch_ratio']].head())


#COMPROBACIONES GRAFICAS PARA ESTADISTICAS (OPCIONAL)
# Visualización para el análisis exploratorio
plt.figure(figsize=(10, 6))
sns.boxplot(x='device_type', y='days_inactive', data=merged_df)
plt.title('Días de Inactividad por Tipo de Dispositivo')
plt.show()

# Análisis estadístico: ANOVA (ya que tenemos más de dos grupos)
# Agrupa los datos de inactividad por tipo de dispositivo
device_groups = [group['days_inactive'] for name, group in merged_df.groupby('device_type')]

# Realiza el ANOVA (podemos considerar otros tipos de analisis pero este es uno de los mas populares)
f_statistic, p_value = stats.f_oneway(*device_groups)

print(f'F-statistic: {f_statistic:.2f}')
print(f'P-value: {p_value:.3f}')

# Interpretación
if p_value < 0.05:
    print("La diferencia en los días de inactividad entre los tipos de dispositivo es estadísticamente significativa.")
else:
    print("No hay una diferencia estadísticamente significativa en los días de inactividad entre los tipos de dispositivo.")


#FASE DEL MODELO DE ML
# Selecciona las características y la variable objetivo
features = ['sessions_per_week', 'unique_genres_watched', 'movie_watch_ratio']
target = 'is_churned_30'

# Elimina las filas con valores faltantes por si los hay
model_df = merged_df[features + [target]].dropna()

X = model_df[features]
y = model_df[target]

# Divide los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Inicializa y entrena el modelo XGBoost
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model.fit(X_train, y_train)

# Evalua el modelo (importante que se repita hasta tener un resultado agradable, pero que sea realista)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#PRUEBAS GRAFICAS
# Crea un "explainer" de SHAP para el modelo evaluado
explainer = shap.TreeExplainer(model)

# Calcula los valores SHAP para el conjunto de prueba
shap_values = explainer.shap_values(X_test)

# Genera el gráfico de resumen de SHAP
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

#FASE DE FILTROS PARA RECOMENDACIONES
# Interacciones agregadas entre usuarios y películas
user_movie_matrix = merged_df.pivot_table(
    index='user_id',
    columns='movie_id',
    values='progress_percentage'
).fillna(0) # Rellena los valores NaN con 0, lo que indica que no hay interacción.

print("User-Item Matrix shape:", user_movie_matrix.shape)
print(user_movie_matrix.head())

# Convierte la matriz usuario-elemento en una matriz dispersa para mayor eficiencia en la compilacion (clave porque consideramos el despliegue en streamlit).
user_movie_sparse = csr_matrix(user_movie_matrix.values)

# Crea el modelo KNN
# Utilizaremos la similitud coseno para medir la distancia entre usuarios
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(user_movie_sparse)


#FASE DE LA FUNCION REAL DE RECOMENDACIONES
def get_movie_recommendations(user_id, num_recommendations=5):
    # Obten el índice del usuario en nuestra matriz
    user_idx = user_movie_matrix.index.get_loc(user_id)
    
    # Encuentra los k vecinos más cercanos (simulando los usuarios similares)
    distances, indices = model_knn.kneighbors(
        user_movie_matrix.iloc[user_idx, :].values.reshape(1, -1),
        n_neighbors=num_recommendations + 1)
        
    # Obten las películas que han visto usuarios similares.
    similar_users_indices = indices.flatten()[1:] # Excluye al propio usuario (evita sesgo)
    similar_users_movies = user_movie_matrix.iloc[similar_users_indices]

    # Busca películas que el usuario objetivo no haya visto
    user_watched_movies = user_movie_matrix.iloc[user_idx][user_movie_matrix.iloc[user_idx] > 0].index.tolist()
    
    # Obtén las películas más recomendadas por usuarios similares.
    recommendations = similar_users_movies.sum().sort_values(ascending=False)
    
    # Filtra las películas que el usuario ya ha visto y obten las N más populares.
    final_recommendations = recommendations[~recommendations.index.isin(user_watched_movies)].head(num_recommendations)
    
    # Obten los detalles de la película del archivo original movies_df
    recommended_movie_ids = final_recommendations.index.tolist()
    recommended_movies = movies_df[movies_df['movie_id'].isin(recommended_movie_ids)]
    
    return recommended_movies[['title', 'genre_primary', 'genre_secondary']]


#FASE DE ARCHIVOS A USAR EN STREAMLIT
# 1. Guarda el modelo de rotación de XGBoost
model_churn = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model_churn.fit(X_train, y_train)
joblib.dump(model_churn, 'churn_model.pkl')

# 2. Guarda el modelo de recomendación KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_knn.fit(user_movie_sparse)
joblib.dump(model_knn, 'knn_model.pkl')

# 3. Guarda los DataFrames y las matrices
user_movie_matrix.to_pickle('user_movie_matrix.pkl')
movies_df.to_csv('movies_df.csv', index=False)
merged_df.to_csv('merged_df.csv', index=False)

# 4. Guarda los valores SHAP (como matrices NumPy)
explainer = shap.TreeExplainer(model_churn)
shap_values = explainer.shap_values(X_test)
np.save('shap_values.npy', shap_values)
X_test.to_csv('X_test.csv', index=False)

