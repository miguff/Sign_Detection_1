#from IPython.display import display, Javascript, Image
import Config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes #ez fel tud ismerni file típusokat filenevek és url-ek alapján
import imutils
import pickle
import cv2
import os
def main():
    input = "/home/miguff/ÓE/Sign_Recognition/test_images/800px_COLOURBOX2900894.jpg"
    #filetype = mimetypes.guess_type(input)
    #print(filetype)

    print("[INFO] loading object detector ... ")
    model = load_model(Config.MODEL_PATH)
    lb = pickle.loads(open(Config.LB_PATH, "rb").read())
    predict_on_image(input, lb, model)

def predict_on_image(input, lb, model):
    image = load_img(input, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis = 0)

    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    image = cv2.imread(input)
    (h, w) = image.shape[:2]
    startX = int(startX*w)
    startY = int(startY*h)
    endX = int(endX*w)
    endY = int(endY*h)
    #print(startX, startY, endX, endY)

    y = startY - 10 if startY -10 >10 else startY +10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
    cv2.imwrite("/home/miguff/ÓE/Sign_Recognition/output/teszt.png", image)

def show_video():
    cam = cv2.VideoCapture(0)
    while True:

        ignore, frame = cam.read()
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (224, 224))
        #image = load_img(frame, target_size=(224, 224))
        image = img_to_array(image)/255.0
        image = np.expand_dims(image, axis=0)

        (boxPreds, labelPreds) = model.predict(image)
        (startX, startY, endX, endY) = boxPreds[0]

        i = np.argmax(labelPreds, axis=1)
        label = lb.classes_[i][0] #Azért kell ide a nulladik, hogy ne listában hanem stringként adja vissza az értéket

        LEVEL = 90
        if max(labelPreds[0])*100 > LEVEL:


        #image = cv2.imread(frame)
        #image = imutils.resize(image, width=600) #itt csak azért kell ez a sor, hogy az eredeti sor beférjen a képernyőre
            (h, w) = image.shape[1:3]
            startX = int(startX*w)
            startY = int(startY*h)
            endX = int(endX*w)
            endY = int(endY*h)
            #print(startX, startY, endX, endY)

            y = startY - 10 if startY -10 >10 else startY +10
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()