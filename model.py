import json
import numpy as np
from PIL import Image, ImageOps
import tensorflowjs as tfjs

def get_class(image_path):
    model = tfjs.converters.load_keras_model('model.json')

    with open('metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        class_names = metadata.get('labels', [])

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction[0])

    class_name = class_names[index] if index < len(class_names) else f"Classe {index}"
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score