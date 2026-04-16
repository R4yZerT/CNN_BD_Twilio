"""
app.py
======
Interfaz Gradio para clasificación de imágenes histológicas.
Usa EfficientNetB0 y envía diagnósticos por SMS vía Twilio.
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf
from twilio.rest import Client
from datetime import datetime
from dotenv import load_dotenv
from tensorflow.keras.applications.efficientnet import preprocess_input

# Cargar variables de entorno (.env)
load_dotenv()

# ── 1. CONFIGURACIÓN ──────────────────────────────────────────────────────────
MODEL_PATH = "models/lung_colon_model.keras"
IMG_SIZE = (224, 224) # Debe coincidir con el entrenamiento

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

CLASS_INFO = {
    "colon_aca": {
        "label": "Adenocarcinoma de Colon",
        "emoji": "🔴",
        "descripcion": "Tejido maligno detectado en colon.",
        "nivel": "ALTO RIESGO",
        "color": "#FF4444",
    },
    "colon_n": {
        "label": "Tejido de Colon Normal",
        "emoji": "🟢",
        "descripcion": "Tejido sin indicios de malignidad.",
        "nivel": "NORMAL",
        "color": "#44BB44",
    },
    "lung_aca": {
        "label": "Adenocarcinoma de Pulmón",
        "emoji": "🔴",
        "descripcion": "Cáncer pulmonar maligno detectado.",
        "nivel": "ALTO RIESGO",
        "color": "#FF4444",
    },
    "lung_n": {
        "label": "Tejido de Pulmón Normal",
        "emoji": "🟢",
        "descripcion": "Tejido pulmonar sano.",
        "nivel": "NORMAL",
        "color": "#44BB44",
    },
    "lung_scc": {
        "label": "Carcinoma Escamoso de Pulmón",
        "emoji": "🔴",
        "descripcion": "Tipo de cáncer de pulmón maligno.",
        "nivel": "ALTO RIESGO",
        "color": "#FF4444",
    }
}

# ── 2. CARGA DEL MODELO ───────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    print(f"✅ Cargando modelo desde {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"❌ ERROR: No se encontró el modelo en {MODEL_PATH}. Entrena primero.")
    model = None

# ── 3. LÓGICA DE PREDICCIÓN ───────────────────────────────────────────────────
def predict(img):
    if model is None:
        return None, "Error: Modelo no cargado", ""

    # Preprocesamiento idéntico al entrenamiento
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    prob = preds[idx]

    info = CLASS_INFO[label]
    
    # Crear HTML para visualización
    html_res = f"""
    <div style='background:{info["color"]}; padding:20px; border-radius:10px; color:white; text-align:center;'>
        <h2>{info["emoji"]} {info["label"]}</h2>
        <p><b>Nivel:</b> {info["nivel"]} | <b>Confianza:</b> {prob:.2%}</p>
        <p>{info["descripcion"]}</p>
    </div>
    """
    
    # Diccionario para la gráfica de Gradio
    confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(5)}
    
    return confidences, html_res, label

# ── 4. LÓGICA DE SMS (TWILIO) ─────────────────────────────────────────────────
def send_sms(predicted_cls, phone_number):
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")

        client = Client(account_sid, auth_token)
        info = CLASS_INFO[predicted_cls]
        
        msg_body = (
            f"Diagnostico IUE (Académico)\n"
            f"Resultado: {info['label']}\n"
            f"Nivel: {info['nivel']}\n"
            f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        message = client.messages.create(
            body=msg_body,
            from_=from_number,
            to=phone_number
        )
        return f"✅ SMS enviado con éxito. SID: {message.sid}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ── 5. INTERFAZ GRADIO ────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    predicted_cls = gr.State()

    gr.HTML("<h1 style='text-align:center;'>🫁 Clasificador Oncológico (Big Data)</h1>")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Subir imagen histológica")
            btn_predict = gr.Button("🔍 Analizar Imagen", variant="primary")
        
        with gr.Column():
            out_chart = gr.Label(label="Probabilidades", num_top_classes=5)
            out_html = gr.HTML()

    with gr.Row():
        with gr.Column():
            phone_input = gr.Textbox(label="Número Twilio (+57...)", placeholder="+573001234567")
            btn_sms = gr.Button("📤 Enviar SMS")
            sms_status = gr.Textbox(label="Estado SMS", interactive=False)

    # Eventos
    btn_predict.click(
        fn=predict, 
        inputs=img_input, 
        outputs=[out_chart, out_html, predicted_cls]
    )
    
    btn_sms.click(
        fn=send_sms, 
        inputs=[predicted_cls, phone_input], 
        outputs=sms_status
    )

if __name__ == "__main__":
    # Desactivamos el api_open para evitar el error de validación de esquemas
    demo.launch(
        share=True, 
        server_name="0.0.0.0", 
        server_port=7860,
        debug=True  # Esto te mostrará el error real en la terminal si algo falla al presionar el botón
    )