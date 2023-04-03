from cProfile import label
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def color_identifier(img):
    model = load_model('c:/Users/sagar/Downloads/fruitvegie/Fruit_Vegetable_Recognition-master/color_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    # image.show()
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    Label = ["Red","Green","Blue","Orange","Yellow","Voilet","Indigo"]
    prediction = list(zip(prediction[0],Label))
    # print("Model Prediction is {0} color.".format(max(prediction)))
    
    return max(prediction)[1]