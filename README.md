# CNN + Twilio Integration Project 🚀

Este proyecto implementa una solución basada en Redes Neuronales Convolucionales (CNN) con integración de mensajería a través de Twilio. Está diseñado para procesar datos/imágenes y disparar notificaciones automatizadas.

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:
* **Python 3.9+**
* **Pip** (gestor de paquetes de Python)
* Una cuenta activa en **Twilio** (para las credenciales de API).

## ⚙️ Instalación

Sigue estos pasos para configurar el entorno de desarrollo:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/R4yZerT/CNN_BD_Twilio.git](https://github.com/R4yZerT/CNN_BD_Twilio.git)
   cd CNN_BD_Twilio
	2	Crear un entorno virtual: En macOS/Linux:Bashpython3 -m venv venv
	3	source venv/bin/activate
	4	En Windows:Bashpython -m venv venv
	5	.\venv\Scripts\activate
	6	
	7	Instalar dependencias:Bashpip install -r requirements.txt
	8	
🔐 Configuración de Variables de Entorno
Este proyecto utiliza un archivo .env para proteger datos sensibles. Crea un archivo llamado .env en la raíz del proyecto y añade lo siguiente:
Fragmento de código

TWILIO_ACCOUNT_SID=tu_sid_aqui
TWILIO_AUTH_TOKEN=tu_token_aqui
TWILIO_PHONE_NUMBER=tu_numero_twilio
TARGET_PHONE_NUMBER=tu_numero_personal

Nota: El archivo .env está incluido en el .gitignore para evitar que tus credenciales se suban a GitHub.
🚀 Ejecución
Para iniciar el entrenamiento del modelo o el servicio principal:
Bash

python main.py
🛠️ Tecnologías utilizadas
	•	TensorFlow/Keras: Para la arquitectura de la red neuronal.
	•	Twilio API: Para la gestión de notificaciones.
	•	Pandas/NumPy: Para el preprocesamiento de datos.
	•	Python-dotenv: Para la gestión de variables de entorno.
