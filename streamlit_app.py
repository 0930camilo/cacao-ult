import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os  # Importación de la biblioteca os

# Diccionario de credenciales de usuario
USER_CREDENTIALS = {
    "yeimer": "password123",
    "user1": "securepass"
}

# Función para manejar el inicio de sesión
def login():
    st.sidebar.title("Inicio de Sesión")
    username = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Contraseña", type="password")
    login_button = st.sidebar.button("Iniciar sesión")

    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.sidebar.success("Inicio de sesión exitoso")
        else:
            st.sidebar.error("Usuario o contraseña incorrectos")
# Inicializar el estado de sesión
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    st.sidebar.button("Cerrar sesión", on_click=lambda: st.session_state.update({"authenticated": False}))

    # Código principal de la aplicación después de iniciar sesión
    st.title("Predicción de Enfermedad")

    # Cargar el modelo previamente entrenado
    model_path = os.path.join(os.getcwd(), "keras_model.h5")
    model = load_model(model_path, compile=False)

    # Cargar las etiquetas
    class_names = open("labels.txt", "r").readlines()

    # Diccionario de recomendaciones basadas en las enfermedades
    recommendations = {
        "Monilia": """
        **Recomendaciones para manejar la Monilia (Monilia spp.) en plantaciones de cacao:**
        
        1. **Eliminación de frutos infectados**: Retira y destruye los frutos afectados por monilia. 
        2. **Mejorar la ventilación y la sombra**: Mantén la plantación bien aireada y con suficiente sombra. 
        3. **Control químico**: Utiliza fungicidas específicos siguiendo las indicaciones del fabricante. 
        4. **Aplicación de productos biológicos**: Usa bacterias o hongos antagonistas (como Trichoderma spp.).
        5. **Poda adecuada**: Realiza podas periódicas para mejorar la circulación del aire. 
        6. **Manejo integrado de plagas y enfermedades (MIPE)**: Combina prácticas culturales, control biológico y químico. 
        7. **Revisión y monitoreo constantes**: Inspecciona regularmente para detectar signos tempranos.
        """,
        "Sana": """
        **Recomendación general para la planta sana:**
        Mantén buenas prácticas de cuidado de la planta, realiza monitoreo regular y asegúrate de mantener condiciones adecuadas de cultivo para prevenir problemas futuros.
        """
    }

    # Subir una imagen a través de la interfaz de usuario de Streamlit
    uploaded_image = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Procesar y redimensionar la imagen cargada para que sea más pequeña
        image = Image.open(uploaded_image).convert("RGB")
        image = image.resize((300, 300))  # Redimensionar a 300x300 píxeles

        # Mostrar la imagen cargada (ahora más pequeña)
        st.image(image, caption="Imagen cargada", use_container_width=False)

        # Redimensionar la imagen para el modelo
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Convertir la imagen a un arreglo numpy
        image_array = np.asarray(image)

        # Normalizar la imagen
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Crear el arreglo para la predicción
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Realizar la predicción
        prediction = model.predict(data)

        # Obtener la clase con la mayor probabilidad
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Eliminar saltos de línea
        confidence_score = prediction[0][index]

        # Mostrar el resultado de la predicción y el puntaje de confianza en porcentaje
        st.subheader(f"Predicción: {class_name}")
        st.write(f"La probabilidad: {confidence_score * 100:.0f}%")

        # Mostrar la recomendación basada en la predicción
        if class_name in recommendations:
            st.write(recommendations[class_name])
        else:
            st.write("No se tiene una recomendación específica para esta enfermedad. Se sugiere consultar a un profesional.")
