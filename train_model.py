import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# RUTAS
BASE_DIR = "/Users/yeipezz/Downloads/lung_colon_cnn/lung_colon_image_set"
COLON_DIR = os.path.join(BASE_DIR, "colon_image_sets")
LUNG_DIR = os.path.join(BASE_DIR, "lung_image_sets")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

def train():
    # 1. GENERADOR CON DATA AUGMENTATION (Para evitar el sobreajuste)
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        horizontal_flip=True,
        validation_split=0.2
    )

    # 2. CARGA DE DATOS POR SEPARADO
    # Colon tiene 2 clases: colon_aca, colon_n
    # Lung tiene 3 clases: lung_aca, lung_n, lung_scc
    
    print("🚀 Cargando datos de Colon...")
    train_colon = datagen.flow_from_directory(
        COLON_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )
    val_colon = datagen.flow_from_directory(
        COLON_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )

    print("🚀 Cargando datos de Pulmón...")
    train_lung = datagen.flow_from_directory(
        LUNG_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )
    val_lung = datagen.flow_from_directory(
        LUNG_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )

    # 3. UNIFICAR GENERADORES (Función helper)
    def combine_gen(gen1, gen2):
        while True:
            x1, y1 = next(gen1)
            x2, y2 = next(gen2)
            # Unimos las imágenes
            x = np.concatenate([x1, x2])
            # Ajustamos las etiquetas para que todas tengan 5 posiciones (2 de colon + 3 de pulmón)
            y1_padded = np.pad(y1, ((0,0), (0,3)), mode='constant')
            y2_padded = np.pad(y2, ((0,0), (2,0)), mode='constant')
            y = np.concatenate([y1_padded, y2_padded])
            yield x, y

    train_final = combine_gen(train_colon, train_lung)
    val_final = combine_gen(val_colon, val_lung)

    # 4. ARQUITECTURA EFFICIENTNETB0
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. ENTRENAMIENTO
    # Calculamos los pasos por época para que el entrenamiento no sea infinito
    steps_per_epoch = (train_colon.samples + train_lung.samples) // (BATCH_SIZE * 2)
    validation_steps = (val_colon.samples + val_lung.samples) // (BATCH_SIZE * 2)

    print("🔥 Entrenando unificado...")
    model.fit(
        train_final,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_final,
        validation_steps=validation_steps,
        epochs=EPOCHS
    )

    # 6. GUARDAR
    os.makedirs("models", exist_ok=True)
    model.save("models/lung_colon_model.keras")
    print("✅ ¡LOGRADO! Modelo guardado en models/lung_colon_model.keras")

if __name__ == "__main__":
    train()