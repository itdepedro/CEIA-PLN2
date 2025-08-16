# Chatbot con Agentes para Búsqueda de CVs 🤖

Este proyecto contiene una solución completa para un chatbot de preguntas y respuestas sobre currículums (CVs) de forma eficiente, utilizando una arquitectura de **Agentes**. La solución se compone de un notebook para la preparación de datos, un script que sintetiza la notebook anterior y un chatbot interactivo construido con **Streamlit**.

-----

## 📁 Estructura del Repositorio
  - **`configurar_vdb.ipynb`**: Script de python que configura un índice en Pinecone, genera embeddings de documentos PDF, y permite realizar búsquedas por similitud.
  - **`clase_7.ipynb`**: Notebook de Jupyter que se encarga de la **preparación y la indexación**. Lee los documentos (CVs), los divide en fragmentos (`chunks`), genera **embeddings** y los carga en una base de datos vectorial de **Pinecone** para su posterior consulta.
  - **`chatbot_gestionada.py`**: Aplicación web interactiva construida con **Streamlit** y **LangChain**. Este chatbot se conecta al índice de Pinecone para recuperar información relevante y utiliza los modelos de lenguaje (LLM) de **Groq** para generar respuestas. La funcionalidad de **memoria** de LangChain permite mantener el contexto de la conversación.

-----

## 🚀 Cómo Empezar

### 1\. Configuración del Entorno

1.  **Clona el repositorio:**
    ```bash
    git clone git@github.com:itdepedro/CEIA-PLN2.git
    ```
2.  **Crea y activa un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```
3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### 2\. Preparar la Base de Datos Vectorial
1.  **Configura tus claves API** como variables de entorno para una mayor seguridad. Puedes hacerlo en tu terminal:
    ```bash
    export PINECONE_API_KEY="tu_clave_pinecone"
    ```
2.  Ejecutar el script `configurar_vdb.py` para configurar y popular la base de datos vectorial.

### 3\. Ejecutar el Chatbot

1.  **Configura tus claves API** como variables de entorno para una mayor seguridad. Puedes hacerlo en tu terminal:
    ```bash
    export PINECONE_API_KEY="tu_clave_pinecone"
    export GROQ_API_KEY="tu_clave_groq"
    ```
2.  Ejecuta la aplicación de Streamlit desde la terminal:
    ```bash
    streamlit run chatbot_gestionada.py
    ```
    El chatbot se abrirá automáticamente en tu navegador, y ya podrás hacer preguntas sobre los documentos que cargaste.