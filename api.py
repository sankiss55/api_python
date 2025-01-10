from flask import Flask, jsonify, request
import easyocr
import cv2
import numpy as np
import os  # Asegúrate de importar el módulo os para acceder a la variable de entorno

app = Flask(__name__)

# Inicializar el lector de EasyOCR
reader = easyocr.Reader(['es', 'en'], gpu=False)

@app.route('/ocr', methods=['POST'])
def ocr_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen.'}), 400
    
    file = request.files['image']

    # Leer la imagen desde el archivo enviado
    try:
        # Convertir la imagen a un array numpy
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

    # Realizar OCR en la imagen
    try:
        results = reader.readtext(image, paragraph=False)
        
        # Convertir el cuadro delimitador a enteros y preparar los datos para JSON
        ocr_data = [
            {'text': res[1], 'confidence': res[2], 'bbox': [[int(i) for i in box] for box in res[0]]} for res in results
        ]
        
        return jsonify({'ocr_results': ocr_data})
    except Exception as e:
        return jsonify({'error': f'Error al realizar OCR: {str(e)}'}), 500

if __name__ == '__main__':
    # Usar la variable de entorno PORT asignada por Railway
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto de Railway, o 5000 como fallback
    app.run(debug=True, host="0.0.0.0", port=port)
