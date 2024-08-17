import os
import streamlit as st
import pandas as pd
from groq import Groq
import torch
import whisper
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import json

# Configura la página de Streamlit para que use todo el ancho disponible
st.set_page_config(layout="wide")

# Establece la clave API para acceder a la API de Groq desde st.secrets
api_key = st.secrets["general"]["GROQ_API_KEY"]

# Inicializa el cliente de Groq usando la clave API
client = Groq(
    api_key = api_key,
)

# Función para obtener respuestas en streaming desde la API
def get_streaming_response(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Función para generar contenido a partir de un modelo Groq
def generate_content(modelo:str, prompt:str, system_message:str="You are a helpful assistant.", max_tokens:int=1024, temperature:int=0.5):
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        model=modelo,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stop=None,
        stream=True
    ) 
    return stream


# Función para transcribir audio usando Whisper
def transcribir_audio(audio):
    # Verificar si la GPU admite FP16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
        fp16_available = True
    else:
        fp16_available = False
    
    model = whisper.load_model("base")

    if fp16_available:
        result = model.transcribe(audio)
    else:
        result = model.transcribe(audio, fp16=False)     
    
    transcripcion = result["text"]
    return transcripcion

# Título de la aplicación Streamlit
st.title("Loope x- 🤖")

# Barra lateral para cargar archivo, seleccionar modelo y ajustar parámetros
with st.sidebar:
    st.write("Estás usando  **Streamlit💻** and **Groq🖥**")
    
    # Permite al usuario subir un archivo Excel
    uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])

    # Permite al usuario subir un archivo de audio
    uploaded_audio = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav", "ogg", "flac"])

    # Permite al usuario seleccionar el modelo a utilizar
    modelo = st.selectbox("Modelo", ["llama3-8b-8192", "mixtral-8x7b-32768", "llama3-70b-8192", "gemma-7b-it"])

    # Permite al usuario ingresar un mensaje de sistema
    system_message = st.text_input("System Message", placeholder="Default : You are a helpful assistant.")
    
    # Ajusta la temperatura del modelo para controlar la creatividad
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.2)
    
    # Selecciona el número máximo de tokens para la respuesta
    max_tokens = st.selectbox("Max New Tokens", [1024, 2048, 4096, 8196])

# Inicializa el historial de chat en el estado de sesión si no existe
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Muestra los mensajes del historial de chat
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Inicializa el estado de sesión si no existe
if "transcripcion_finalizada" not in st.session_state:
    st.session_state["transcripcion_finalizada"] = False
if "transcripcion" not in st.session_state:
    st.session_state["transcripcion"] = ""


# Si se ha cargado un archivo de audio, lo transcribe y envía la transcripción al modelo
if uploaded_audio is not None:
    st.write("Transcribiendo el audio...")
    transcripcion = transcribir_audio(uploaded_audio)
    st.write("Transcripción del audio:")
    st.write(transcripcion)
    
    # Envía la transcripción al modelo para corrección gramatical y asignación de interlocutores
    prompt = f"Corrige gramaticalmente el siguiente texto y asigna interlocutores:\n\n{transcripcion}"
    
    st.session_state["chat_history"].append(
        {"role": "user", "content": prompt},
    )
    with st.chat_message("user"):
        st.write(prompt)
    
    response = generate_content(modelo, prompt, system_message, max_tokens, temperature)
    
    with st.chat_message("assistant"):
        stream_generator = get_streaming_response(response)
        streamed_response = st.write_stream(stream_generator)
    
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": streamed_response},
    )

# Si se ha cargado un archivo Excel, procesa y muestra su contenido
if uploaded_file is not None:
    # Carga el archivo Excel en un DataFrame
    dfs = pd.read_excel(uploaded_file)
    
    # Convertir columnas de texto a tipo str
    df = dfs.astype({col: str for col in dfs.select_dtypes(include=['object']).columns})
    
    # Muestra el contenido del archivo en la interfaz
    st.write("Contenido del archivo:")
    st.dataframe(df.head(3))

    # Convierte columnas de datetime a str si existen
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    lista_diccionario = df.to_dict(orient="records")
    lista_diccionario_texto = json.dumps(lista_diccionario, ensure_ascii=False, indent=2)    

    llm = ChatGroq(model_name=modelo, api_key = api_key)
    # Inicializa SmartDataframe
    smart_df = SmartDataframe(dfs, config={'llm': llm})
    
    # Solicita preguntas separadas para cada barra de chat
    #col1, col2 = st.columns(2)

    
    prompt_pandasai = st.chat_input("Haz una petición para el archivo (PandasAI)...")
    
    prompt_dict = st.chat_input("Haz una pregunta sobre el archivo (Diccionario)...")

    if prompt_pandasai:
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt_pandasai},
        )
        with st.chat_message("user"):
            st.write(prompt_pandasai)
        
        # Solicita explícitamente código Python en la respuesta
        code_prompt = f"Genera el código Python necesario para resolver el siguiente problema:\n\n{prompt_pandasai}"
        response_pandasai = smart_df.chat(code_prompt)
        
        with st.chat_message("assistant"):
            st.write(response_pandasai)
        
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response_pandasai},
        )

    if prompt_dict:
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt_dict},
        )
        with st.chat_message("user"):
            st.write(prompt_dict)
        
        response_prompt = f"{prompt_dict}\n\nDatos del archivo:\n{lista_diccionario_texto}"
        response = generate_content(modelo, response_prompt, system_message, max_tokens, temperature)
        
        with st.chat_message("assistant"):
            stream_generator = get_streaming_response(response)
            streamed_response = st.write_stream(stream_generator)
        
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": streamed_response},
        )
# Si no se ha cargado un archivo, permite hacer preguntas generales
if uploaded_file is None and uploaded_audio is None:
    prompt = st.chat_input("Haz una pregunta general...")

    if prompt:
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt},
        )
        with st.chat_message("user"):
            st.write(prompt)
        
        response = generate_content(modelo, prompt, system_message, max_tokens, temperature)
        
        with st.chat_message("assistant"):
            stream_generator = get_streaming_response(response)
            streamed_response = st.write_stream(stream_generator)
        
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": streamed_response},
        )
