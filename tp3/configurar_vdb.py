"""
# CEIA-PLN2/tp3/configurar_vdb.py
# Este script configura un índice en Pinecone, genera embeddings de documentos PDF,
# y permite realizar búsquedas por similitud.

# Requiere las siguientes variables de entorno:
# - PINECONE_API_KEY: Clave API de Pinecone
# - PINECONE_ENVIRONMENT: Entorno de Pinecone (ej: 'us-west1-gcp')

"""

# ========================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ========================================
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, PodSpec
from pinecone import ServerlessSpec
import re
import pdfplumber

# ========================================
# CONFIGURACIÓN DE PINECONE Y GROQ
# ========================================
nombre_indice = "tp3-pln2"
PINECONE_ENVIRONMENT = "us-west1-gcp"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ========================================
# EXTRACCIÓN DE TEXTO DESDE PDF
# ========================================
def extraer_texto_pdf(ruta_pdf):
    texto = ""
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto += pagina.extract_text() + "\n"
    return texto


# ================================
# 1. CONFIGURACIÓN INICIAL
# ================================

def configurar_pinecone():
    """
    Configura la conexión con Pinecone usando variables de entorno.
    
    Variables necesarias:
    - PINECONE_API_KEY: Tu clave API de Pinecone
    - PINECONE_ENVIRONMENT: El entorno de Pinecone (ej: 'us-west1-gcp')
    """
    
    # Obtener credenciales desde variables de entorno
    api_key = PINECONE_API_KEY
    environment = PINECONE_ENVIRONMENT
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY no está configurada en las variables de entorno")
    
    # Inicializar Pinecone
    pc = Pinecone(api_key=api_key)
    
    print(f"✅ Pinecone configurado correctamente en {environment}")
    return True

class GeneradorEmbeddings:
    """
    Clase para generar embeddings usando diferentes modelos.
    """
    
    def __init__(self, modelo: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el generador de embeddings.
        
        Args:
            modelo (str): Nombre del modelo de Sentence Transformers
        """
        self.modelo_nombre = modelo
        self.modelo = SentenceTransformer(modelo)
        self.dimension = self.modelo.get_sentence_embedding_dimension()
        
        print(f"✅ Modelo '{modelo}' cargado (dimensión: {self.dimension})")
    
    def generar_embedding(self, texto: str) -> List[float]:
        """
        Genera embedding para un texto individual.
        
        Args:
            texto (str): Texto a convertir en embedding
            
        Returns:
            List[float]: Vector de embedding
        """
        embedding = self.modelo.encode(texto)
        return embedding.tolist()
    
    def generar_embeddings_lote(self, textos: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos de manera eficiente.
        
        Args:
            textos (List[str]): Lista de textos
            
        Returns:
            List[List[float]]: Lista de vectores de embedding
        """
        embeddings = self.modelo.encode(textos)
        return [emb.tolist() for emb in embeddings]
def crear_indice(nombre_indice: str, dimension: int = 384, metrica: str = "cosine"):
    """
    Crea un nuevo índice en Pinecone.
    
    Args:
        nombre_indice (str): Nombre del índice a crear
        dimension (int): Dimensión de los vectores (depende del modelo de embedding)
        metrica (str): Métrica de similitud ('cosine', 'euclidean', 'dotproduct')
    
    Configuración de infraestructura:
        - Pods: Unidades de cómputo paralelo que procesan las consultas
          • 1 pod = suficiente para desarrollo y proyectos pequeños
          • Más pods = mayor capacidad de consultas simultáneas pero mayor costo
        
        - Réplicas: Copias idénticas del índice distribuidas geográficamente
          • 1 réplica = configuración básica
          • Más réplicas = mayor disponibilidad y tolerancia a fallos
        
        - Tipos de pod disponibles:
          • p1.x1: 1 vCPU, ~5GB RAM (plan gratuito/starter)
          • p1.x2: 2 vCPU, ~10GB RAM
          • p1.x4: 4 vCPU, ~20GB RAM
          • p2.x1: Optimizado para performance
    
    Returns:
        bool: True si se creó exitosamente
    """
    
    # # Verificar si el índice ya existe
    # indices_existentes = pinecone.list_indexes().names()
    
    # if nombre_indice in indices_existentes:
    #     print(f"⚠️  El índice '{nombre_indice}' ya existe")
    #     return True
    

    pc = Pinecone(api_key=PINECONE_API_KEY)
   
        
    # Crear el índice
    if nombre_indice not in pc.list_indexes().names():
        
        pc.create_index(
            name=nombre_indice,
            dimension=dimension,
            metric=metrica,
            spec=ServerlessSpec(
                cloud='aws',
                region="us-east-1"
            )
            # pods=1,  # Pods: Unidades de cómputo que procesan queries. Más pods = mayor throughput pero mayor costo
            # replicas=1,  # Réplicas: Copias del índice para alta disponibilidad. Más réplicas = mayor disponibilidad
            # pod_type="p1.x1"  # Tipo de pod: p1.x1 (gratuito, 1 vCPU), p1.x2 (2 vCPU), p2.x1 (optimizado), etc.
        )
    
        
    else:
        print(f"ℹ️  El índice '{nombre_indice}' ya existe, no se creará uno nuevo")
        return True
    
    # Esperar a que el índice esté listo
    print(f"🔄 Creando índice '{nombre_indice}'...")
    # while nombre_indice not in pinecone.list_indexes():
    #     time.sleep(1)
    
    print(f"✅ Índice '{nombre_indice}' creado exitosamente")
    return True

# ================================
# 3. POBLACIÓN DEL ÍNDICE
# ================================

def poblar_indice_ejemplo(nombre_indice: str, generador: GeneradorEmbeddings,docs):
    """
    Puebla el índice con datos de ejemplo.
    
    Args:
        nombre_indice (str): Nombre del índice de Pinecone
        generador (GeneradorEmbeddings): Instancia del generador de embeddings
    """
   
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,chunk_overlap=20,
            length_function=len
        )
    
    
    # Preparar datos para inserción en lotes
    vectors_para_insertar = []
        
    # Conectar al índice
    indice = pc.Index(nombre_indice)
    id = 1
    for doc in docs:
        # Leer el texto del documento
        print(f"📄 Procesando documento: {doc['cv_file']}")
        texto_cv = extraer_texto_pdf(doc['cv_file'])
        # Reemplazar los saltos de línea con un espacio
        texto_cv= texto_cv.replace('\n', ' ')
        texto_cv= texto_cv.replace('\n\n', ' ')
        texto_cv = re.sub(r'\s+', ' ', texto_cv).strip()
        print(f"Longitud del texto {len(texto_cv)}")
        # Generar los chunks del texto
        print("🔄 Generando chunks del texto...")
        
        chunks = text_splitter.create_documents([texto_cv])
    
        print(f"🔄 Poblando índice con {len(chunks)} chunks...")
        
        # Generar embeddings para todos los textos
        textos = [chunk.page_content for chunk in chunks]
        embeddings = generador.generar_embeddings_lote(textos)
    
        
        for i, chunk in enumerate(textos):
            vector_data = {
                "id": str(id),
                "values": embeddings[i],
                "metadata": {
                    "texto": str(chunk),  # Aquí puedes agregar más metadata si es necesario
                    "nombre": doc['nombre']+" "+doc['apellido']
                }
            }
            vectors_para_insertar.append(vector_data)
            id+=1
    
    # Insertar vectores en el índice
    indice.upsert(vectors=vectors_para_insertar)
    
    # Verificar estadísticas del índice
    estadisticas = indice.describe_index_stats()
    print(f"✅ Índice poblado exitosamente")
    print(f"   📊 Total de vectores: {estadisticas['total_vector_count']}")
    print(f"   📏 Dimensión: {estadisticas['dimension']}")
    
    return True

# ================================
# 4. BÚSQUEDAS EN EL ÍNDICE
# ================================

def buscar_documentos_similares(
    nombre_indice: str, 
    consulta: str, 
    generador: GeneradorEmbeddings,
    top_k: int = 3,
    filtro_metadata: Dict = None
) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda por similitud en el índice.
    
    Args:
        nombre_indice (str): Nombre del índice de Pinecone
        consulta (str): Texto de consulta para buscar
        generador (GeneradorEmbeddings): Generador de embeddings
        top_k (int): Número de resultados más similares a devolver
        filtro_metadata (Dict): Filtros opcionales por metadata
        
    Returns:
        List[Dict]: Lista de documentos similares con scores
    """
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Conectar al índice
    indice = pc.Index(nombre_indice)
    
    # Generar embedding para la consulta
    print(f"🔍 Buscando documentos similares a: '{consulta}'")
    embedding_consulta = generador.generar_embedding(consulta)
    # Realizar la búsqueda
    resultados = indice.query(
        vector=embedding_consulta,
        top_k=top_k,
        include_metadata=True,
        filter=filtro_metadata
    )
    
    # Procesar y formatear resultados
    documentos_encontrados = []
    
    print(f"\n📋 Resultados encontrados ({len(resultados['matches'])}):")
    print("=" * 80)
    
    for i, match in enumerate(resultados['matches'], 1):
        documento = {
            "posicion": i,
            "id": match["id"],
            "score": round(match["score"], 4),
            "texto": match["metadata"]["texto"],
        }
        
        documentos_encontrados.append(documento)
        
        # Mostrar resultado formateado
        print(f"{i}. ID: {documento['id']}")
        print(f"   📊 Score: {documento['score']}")
        print(f"   📝 Texto: {documento['texto'][:100]}...")
        print("-" * 80)
    
    return documentos_encontrados

# ================================
# 5. GESTIÓN DEL ÍNDICE
# ================================

def obtener_estadisticas_indice(nombre_indice: str):
    """
    Muestra estadísticas detalladas del índice.
    
    Args:
        nombre_indice (str): Nombre del índice
    """
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Conectar al índice
    indice = pc.Index(nombre_indice)
    estadisticas = indice.describe_index_stats()
    
    print(f"\n📊 ESTADÍSTICAS DEL ÍNDICE '{nombre_indice}'")
    print("=" * 50)
    print(f"📦 Total de vectores: {estadisticas.get('total_vector_count', 0)}")
    print(f"📏 Dimensión: {estadisticas.get('dimension', 0)}")
    
    # Mostrar estadísticas por namespace si existen
    if 'namespaces' in estadisticas:
        print(f"🏷️  Namespaces:")
        for namespace, stats in estadisticas['namespaces'].items():
            print(f"   - {namespace}: {stats.get('vector_count', 0)} vectores")

def limpiar_indice_completo(nombre_indice: str):
    """
    Elimina todos los vectores del índice.
    
    Args:
        nombre_indice (str): Nombre del índice a limpiar
    """
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Conectar al índice
    indice = pc.Index(nombre_indice)
    
    print(f"🧹 Limpiando índice '{nombre_indice}' completamente...")
    indice.delete(delete_all=True)
    
    print("✅ Índice limpiado exitosamente")


def main():
    """
    Ejecuta un ejemplo completo de uso de Pinecone:
    1. Configuración
    2. Creación del índice
    3. Población con datos
    4. Búsquedas de ejemplo
    5. Limpieza (opcional)
    """
    docs = [
        {
            "cv_file":"cv_de_pedro_es.pdf",
            "nombre": "Ignacio",
            "apellido": "de Pedro"
        },
        {
            "cv_file":"cv_topp.pdf",
            "nombre": "Alejandro",
            "apellido": "Topp"
        },
        {
            "cv_file":"cv_de_pedro_f.pdf",
            "nombre": "Francisco",
            "apellido": "de Pedro"
        }
    ]

    try:
        print("🚀 INICIANDO EJEMPLO DE PINECONE")
        print("=" * 50)
        
        # 1. Configurar conexión
        configurar_pinecone()
        
        # 2. Inicializar generador de embeddings
        generador = GeneradorEmbeddings()
        
        # 3. Crear índice
        crear_indice(nombre_indice, dimension=generador.dimension)
        
        # 4. Poblar índice con datos de ejemplo
        poblar_indice_ejemplo(nombre_indice, generador,docs=docs)
        
        # # 5. Mostrar estadísticas
        # obtener_estadisticas_indice(nombre_indice)
        
        # # 6. Realizar búsquedas de ejemplo
        # buscar_con_filtros_ejemplo(nombre_indice, generador)
        
        # 7. Búsqueda personalizada
        print("\n🎯 BÚSQUEDA PERSONALIZADA - Ignacio de Pedro")
        print("=" * 30)
        consulta_personalizada = "Nacionalidad"
        filtro_nombre = {"nombre": {"$eq": "Ignacio de Pedro"}}
        resultados = buscar_documentos_similares(
            nombre_indice, 
            consulta_personalizada, 
            generador,
            top_k=2,
            filtro_metadata=filtro_nombre
        )
        
        # 7. Búsqueda personalizada
        print("\n🎯 BÚSQUEDA PERSONALIZADA - Alejandro Topp")
        print("=" * 30)
        consulta_personalizada = "Nacionalidad"
        filtro_nombre = {"nombre": {"$eq": "Alejandro Topp"}}
        resultados = buscar_documentos_similares(
            nombre_indice, 
            consulta_personalizada, 
            generador,
            top_k=2,
            filtro_metadata=filtro_nombre
        )
        
                
        print(f"\n✅ EJEMPLO COMPLETADO EXITOSAMENTE")
        print(f"📁 Índice '{nombre_indice}' está listo para usar")
        
        # Opcional: Comentar la siguiente línea si quieres mantener el índice
        # eliminar_indice(nombre_indice)
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        raise


if __name__ == "__main__":
    # Punto de entrada de la aplicación
    # Solo ejecutar main() si este archivo se ejecuta directamente
    main()
