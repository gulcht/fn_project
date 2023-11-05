from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def preprocess_image(image_file):
    try:
        img = image.load_img(image_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array.reshape(1, 224, 224, 3)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error while processing image: {e}")
        return None
