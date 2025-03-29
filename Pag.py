
import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cargar modelo de superresolución de OpenCV (ESPCN x4)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("ESPCN_x4.pb")  # Modelo preentrenado
sr.setModel("espcn", 4)  # Escala de 4x

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Subir imagen
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Mejorar la calidad de la imagen
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = sr.upsample(img)  # Aplicar superresolución

        # Guardar la imagen mejorada
        output_path = os.path.join(OUTPUT_FOLDER, "enhanced_" + filename)
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        return render_template("index.html", uploaded_image=filename, output_image="enhanced_" + filename)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

# Configuración para Render: Flask debe escuchar en el puerto correcto
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render asigna un puerto automáticamente
    app.run(host="0.0.0.0", port=port)
