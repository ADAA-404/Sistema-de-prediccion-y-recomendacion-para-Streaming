# Sistema de predicción y recomendación de tipo Netflix 🍿

Este proyecto presenta un proceso integral de ciencia de datos para un servicio de streaming ficticio, centrándose en dos objetivos comerciales fundamentales: predecir la pérdida de usuarios y mejorar su participación mediante un sistema personalizado de recomendación de películas. Demuestra la capacidad de gestionar todo el ciclo de vida de la ciencia de datos, desde la limpieza de datos y la ingeniería de características hasta la implementación de modelos y la explicabilidad. 

## Fuente de Datos 💾

El proyecto utiliza un conjunto de datos sintéticos obtenidos publicamente de Kaggle. Los datos se dividen en varios archivos CSV, pueden ser descargados y consultados aqui

> https://www.kaggle.com/datasets/sayeeduddin/netflix-2025user-behavior-dataset-210k-records?resource=download

## Tecnologias usadas 🐍
Este proyecto se desarrolló utilizando un sólido conjunto de bibliotecas y herramientas de Python para cubrir todo el ciclo de vida del script.  

-   Pandas y NumPy: para la limpieza, manipulación y ingeniería de características de los datos.
-   Scikit-learn: para la división de datos, la evaluación de modelos y la implementación del sistema de recomendaciones basado en KNN.
-   XGBoost: una biblioteca de refuerzo de gradientes de alto rendimiento utilizada para construir el modelo de predicción de abandono.
-   SHAP (SHapley Additive exPlanations): Una potente biblioteca para explicar los resultados de los modelos de aprendizaje automático, fundamental para comunicar información empresarial.
-   Matplotlib y Seaborn: para la visualización de datos, incluidos gráficos estadísticos y los resultados de SHAP.
-   Streamlit: para crear e implementar el panel web interactivo, que presenta los resultados del proyecto en un formato fácil de usar.
-   Joblib: para guardar y cargar de manera eficiente los modelos de aprendizaje automático entrenados, lo que agiliza la aplicación.

## Consideraciones en Instalación ⚙️

Para configurar y ejecutar este proyecto, se recomienda utilizar un entorno `conda`. Estas librerias te ayudarán a crear el entorno necesario:

bash
    ```
    pip install pandas numpy scikit-learn XGBoost seaborn matplotlib SHAP Streamlit Joblib
    ```  
    
Configuración de Datos: Asegúrate de que los archivos de datos de la base de datos (conseguidas por Kaggle) esten ubicados en la carpeta con la que trabajas dentro de la estructura del proyecto.  
Ejecuta el Script: Simplemente corre el script principal para garantizar el funcionamientos desde cero, o despliega el streamlit considerando que tengas los archivos adjuntados en el repositorio para que funcione con la versión base.

## Ejemplo de Uso 📎

El pipeline de La solución es un proceso multifásico creado con Python (probado desde Jupyter Notebook) e implementado con Streamlit.

Ingeniería de datos y creación de características:
-   Consolidación de datos: los tres archivos CSV se fusionaron en un único DataFrame unificado.  
-   Definición de abandono: se definió como un periodo de inactividad de 30, 60 o 90 días.  
-   Métricas de interacción: para cuantificar la interacción de los usuarios, entre ellas total_watch_duration, sessions_per_week y unique_genres_watched.  

![Compilacion del script para DataFrame](Images/Ing_datos.png)

Modelo de predicción de abandono:
-   Selección del modelo: se eligió un clasificador XGBoost por su alto rendimiento y su capacidad para manejar datos estructurados.  
-   Rendimiento del modelo: el modelo alcanzó una precisión de 0,82 y una alta recuperación de 0,92 (se puede comprobar mediante mas procedimiento).   

![Compilacion del script para modelo de prediccion](Images/predict_model.png)

Explicabilidad del modelo con SHAP:
-   Idea clave: Se utilizó SHAP (SHapley Additive exPlanations) donde el análisis reveló que sessions_per_week y unique_genres_watched eran los dos factores más importantes para predecir la pérdida de clientes.  

![Compilacion del script para el modelo shap](Images/Shap_model.png)

Sistema de recomendación de películas"
-   Metodología: Se implementó un enfoque de filtrado colaborativo utilizando K-Nearest Neighbors (KNN).    
-   Matriz usuario-elemento: La participación de los usuarios se cuantificó mediante una métrica de porcentaje de progreso, que se utilizó para construir una matriz usuario-elemento dispersa para el modelo KNN.  
-   Resultado: El sistema recomienda con éxito películas a los usuarios basándose en los hábitos de visualización de usuarios similares, lo que proporciona una experiencia más personalizada.  

![Compilacion del script para recomendacion fianl](Images/Recco_Syst.png)

Implementación del panel de control:
-   Todo el proyecto se implementa como una aplicación web interactiva utilizando Streamlit.

![Compilacion del script para desplegar streamlit](Images/Streamlit_deploy.png)


## Contribuciones 🖨️

Si te interesa contribuir a este proyecto o usarlo independiente, considera:
-   Hacer un "fork" del repositorio.
-   Crear una nueva rama (`git checkout -b feature/su-caracteristica`).
-   Realizar tus cambios y "commiteelos" (`git commit -am 'Agrega nueva característica'`).
-   Subir los cambios a la rama (`git push origin feature/su-caracteristica`).
-   Abrir un "Pull Request".

## Licencia 📜

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE (si aplica) para más detalles.


[English Version](README.en.md)

