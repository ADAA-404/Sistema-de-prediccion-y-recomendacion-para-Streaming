#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Carga datos y modelos (se intento hacer lo mejor optimizado posible) ---
# Considera revisar los archivos del script base para el merged_df completo,
# Revisa el modelo de abandono entrenado y el modelo de recomendación KNN.

@st.cache_data
def load_assets():
    """Carga todos los archivos de datos y modelos para el dashboard."""
    try:
        merged_df = pd.read_csv('merged_df.csv')
        movies_df = pd.read_csv('movies_df.csv')
        user_movie_matrix = pd.read_pickle('user_movie_matrix.pkl')
        X_test = pd.read_csv('X_test.csv')
        
        model_churn = joblib.load('churn_model.pkl')
        model_knn = joblib.load('knn_model.pkl')
        
        # Para SHAP, cargamos los valores y el conjunto de datos de prueba
        shap_values = np.load('shap_values.npy', allow_pickle=True)

    except FileNotFoundError as e:
        st.error(f"Error: {e}. Asegúrate de que todos los archivos de datos y modelos estén en la misma carpeta que este script.")
        st.stop() # Detiene la ejecución si falta un archivo

    return merged_df, movies_df, user_movie_matrix, model_churn, model_knn, shap_values, X_test

# Carga todos los activos del proyecto
merged_df, movies_df, user_movie_matrix, model_churn, model_knn, shap_values, X_test = load_assets()


# --- 2. FUNCIÓN DE RECOMENDACIÓN DINÁMICA ---
def get_movie_recommendations(user_id, num_recommendations=5):
    """Genera recomendaciones reales usando el modelo KNN cargado."""
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    user_idx = user_movie_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(
        user_movie_matrix.iloc[user_idx, :].values.reshape(1, -1),
        n_neighbors=num_recommendations + 1)
        
    similar_users_indices = indices.flatten()[1:]
    similar_users_movies = user_movie_matrix.iloc[similar_users_indices]

    user_watched_movies = user_movie_matrix.iloc[user_idx][user_movie_matrix.iloc[user_idx] > 0].index.tolist()
    
    recommendations = similar_users_movies.sum().sort_values(ascending=False)
    
    final_recommendations = recommendations[~recommendations.index.isin(user_watched_movies)].head(num_recommendations)
    
    recommended_movie_ids = final_recommendations.index.tolist()
    
    # ACTUALIZACION AQUÍ: Añade 'content_type' a las columnas seleccionadas, es una agregado estetico (muy recomendado)
    recommended_movies = movies_df[movies_df['movie_id'].isin(recommended_movie_ids)]
    
    # Asegúrate de que 'content_type' exista en movies_df antes de intentar seleccionarla
    if 'content_type' in recommended_movies.columns:
        return recommended_movies[['title', 'genre_primary', 'genre_secondary', 'content_type']]
    else:
        # En caso de que movies_df no tenga 'content_type' (si el archivo movies.csv no la tenía)
        st.warning("La columna 'content_type' no se encontró en los datos de películas.")
        return recommended_movies[['title', 'genre_primary', 'genre_secondary']]

# --- 3. CONSTRUCCIÓN DE LA INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Netflix Dashboard", layout="wide")
st.title("🎬 Netflix Churn Prediction & Recommendation Dashboard")

# --- Sección de Churn Model Insights ---
st.header("1. Churn Model Insights")
st.markdown("Este dashboard interactivo muestra los **factores clave** que predicen el abandono de usuarios y un **sistema de recomendación** personalizado.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Importancia de las Características (SHAP)")
    st.write("El siguiente gráfico muestra qué tan importante es cada característica para la predicción del modelo.")

    #DEBUG para no batallar con las graficas en el despliegue
    # Crea una NUEVA figura para el gráfico de barras SHAP
    # 1. Crea una figura nueva y vacía
    plt.figure()
    # 2. Genera el gráfico SHAP de barras (sin pasar 'ax')
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    # 3. Pasa la figura actual a Streamlit
    st.pyplot(plt.gcf())
    # 4. Cierra la figura para evitar que se guarde o se reutilice en otro lugar
    plt.close()

with col2:
    st.subheader("Análisis Detallado de SHAP")
    st.write("Cada punto representa un usuario, mostrando cómo el valor de una característica (eje X) influye en la predicción de churn (color).")

    #DEBUG para no batallar con las graficas en el despliegue
    # 1. Crea una figura nueva
    plt.figure()
    # 2. Genera el gráfico SHAP de puntos
    shap.summary_plot(shap_values, X_test, show=False)
    # 3. Pasa la figura actual a Streamlit
    st.pyplot(plt.gcf())
    # 4. Cierra la figura
    plt.close()

st.divider()

# --- Sección de Sistema de Recomendación ---
st.header("2. Movie Recommendation System")
st.write("Obtén recomendaciones de películas personalizadas para cualquier usuario de la plataforma.")

# Usa un selectbox para que el usuario elija un ID válido (Aun que estaria bien mejorar el tipo de busqueda)
sample_users = user_movie_matrix.index.tolist()
user_input = st.selectbox("Selecciona un User ID", sample_users)

if st.button("Obtener Recomendaciones"):
    with st.spinner('Generando recomendaciones...'):
        recommendations = get_movie_recommendations(user_input, num_recommendations=5)
    
    if not recommendations.empty:
        st.success(f"✅ ¡Recomendaciones generadas para el usuario **{user_input}**!")
        st.write("Aquí están las 5 mejores películas recomendadas para ti:")
        st.dataframe(recommendations.style.set_properties(**{'font-size': '12pt'}))
    else:
        st.error("❌ ID de usuario no válido o no se encontraron recomendaciones. Intenta con otro.")


# In[ ]:




