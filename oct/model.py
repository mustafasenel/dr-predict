import cv2


from keras.models import load_model
model = load_model("oct/oct-model.h5")

def preprocess(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,(160,160))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def crop(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,(400,400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image