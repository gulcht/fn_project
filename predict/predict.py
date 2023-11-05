import pickle
import numpy as np

pickled_model = pickle.load(open('weather_model.pkl', 'rb'))
# class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
class_names = ['dew','fogsmog','frost','glaze','hail','lightning',
               'rain','rainbow','rime','sandstorm','snow']

def predict(processed_image):
    prediction = pickled_model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class
