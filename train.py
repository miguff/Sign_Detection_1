import Config
from tensorflow.keras.applications import VGG16 #Ez a CNN architektúra amit használni és módosítani fodgunk
from tensorflow.keras.layers import Flatten #A következők adott szintek a neurális hálózatban
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split #Ennek a segítségével kell kettévenni azokat a képeket amik a trainingre mennek és azokat amik testre
from imutils import paths
import matplotlib.pyplot as plt #Grafikun ábrázolás
import numpy as np
import pickle
import cv2
import os

data = []
labels = []
bboxes = []
imagePaths = []

#Most elkezdjük megtölteni az előző 4 listát külnböző adatokkal
for csvPath in paths.list_files(Config.ANNOTS_PATH, validExts=(".csv")): #Itt az előbb beállított annotációs helyről begyűjti azokat a fileokat amiknek a kiterjestése .csv
    rows = open(csvPath).read().strip().split('\n') #Majd végig megy a sorokon és enterenként elválasztja őket
    #print(rows)
    for row in rows: #Itt azon a listán végigmegyünk amit elpbb létrehoztunk
        row = row.split(',') #Majd amik benne vannak azokat a vesszőknél külön szedjük
        (filename, startX, startY, endX, endY, label) = row #és végül amik külön lettek szedve , azokat külön változóhoz rendeljük
        imagePath = os.path.sep.join([Config.IMAGES_PATH, filename])

        #print(imagePath)
        image = cv2.imread(imagePath) #Ezzel beolvassuk a képet
        (h, w) = image.shape[:2] #Ezzel kiszedjük, hogy mekkora a magassága a képnek és mekkora a szélessége

        #Most méretezzük a bounding box-okat a térbeli méretéhez a bemeneti képhez (ez egy 0-1 közötti értéket ad vissza)
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        #Itt betöltjük a képeket, és átméretezzük őket 224*224*3-assá
        image = load_img(imagePath, target_size=(224, 224)) #Ez azért kell, hogy a most használatos hálózatba beleférjen
        image = img_to_array(image) #Majd numpy array-é tesszük


        #Itt betöltjük a listákba az adatoka
        data.append(image) #a data-ba a képet numpy array-ként
        labels.append(label) #A labels-be a hozzá tartozó labelt
        bboxes.append((startX, startY, endX, endY)) #A bboxes-ba betöltjük a bounding box átméretezett koordinátáit
        imagePaths.append(imagePath) #Az imagePaths-be pedig betöltjük a képeknek a helyét


#Minden adatot numpy array-é konvertálunk, hogy egyszerűbben tudjunk velünk
#számolni, illetve méretezzük a bemeneti pixeleket, hogy 0 és 1 között legyenek

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype = "float32")
imagePaths = np.array(imagePaths)


#one-hot encodingot csinálunk, azaz, úgy alakítjuk át, hogy mindig az egyiket preferálja csak
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#Ha csak 2 osztály van akkor elég a kategórikus is
if len(lb.classes_) == 2:
    labels = to_categorical(labels)

#ezt az utolsó 2 lépést elvileg már megtudnánk csinálni magunktól is, szóval majd
#Olvassuk újra könyvet meg az adatokat és megnézzük hogyan

#Most szétosztjuk 80-20 arányban a train - test egységeket
split = train_test_split(data, labels, bboxes, imagePaths,
                         test_size = 0.2, random_state = 42)
#Ez most szétosztotta, alapból 4 változó volt, 0,1,2,3 értékkel,
#A szétosztás után 8 változó lett, 0,1,2,3,4,5,6,7
#Ebből a 0 és 1, a data lesz szétosztva 80-20-ban, a 2,3 pedig a labels 80-20-ban, és stb
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

print("[Info] saving testing image paths ...")
f = open(Config.TEST_PATHS, "w") #Megnyitjuk azt a file-t amit az elején létroztunk txt-t és abba írjuk bele a a testnke a helyét
f.write("\n".join(testPaths))
f.close()

#Betöltjuk a VGG16 hálózatot, de kihagyjuk belőle a
#vezető fully-connected szintet
vgg = VGG16(weights="imagenet", include_top=False,
            input_tensor = Input(shape=(224, 224, 3))) #ezzel beállítottuk neki, hogy milyen legyen a bemeneti értk

#Minden szintet megfagyasztunk, hogy ezek ne update-elődjenek a training process folyamán
vgg.trainable = False

#Itt pedig 'kiegyenesítjuk az kimenő értékeit a max-pooling kimenetét
flatten = vgg.output
flatten = Flatten()(flatten)#Ezt a Layert fogjuk továbbvinni

#Megépítjük a hálózatunk teljesen kapcsolt 'fejét', ez fogja megtippelni a
# Bounding boxokat

#Itt a szintek 128, 54, 31, 4 neuronból/node-ból állnak
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name = "bounding_box")(bboxHead)
#Az utolsó azért 4 mert az a 4 neuron lesz felelős a 4 x,y koordinátáéert

#Itt megcsináljuk a második teljesen kapcsolt réteget, csak itt a class label-t fogja
#megmondani

softmaxHead = Dense(512, activation='relu')(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation='relu')(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation='softmax', name = "class_label")(softmaxHead)

#Itt egyberakjuk a modellünket, ami bevesz egy értéket és visszaad egy bounding boxot és egy osztály címkét

model = Model(
    inputs=vgg.input,
    outputs = (bboxHead, softmaxHead)
)



#létrehozunk egy szótárat, hogy abba tároljuk el, hogy melyik layert milyen
#Loss funkcióval szeretnénk korrigálni

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error"
}
#Megadjuk, hogy mennyire súlyozza őket, és azt szeretnénk, hogy egyenlően
#legyenek súlyozva

lossWeights = {
    "class_label" : 1.0,
    "bounding_box" : 1.0
}

#megcsináljuk az optimalizálót, ami most egy Adam optimalizáló
opt = Adam(learning_rate=Config.INIT_LR)
model.compile(loss = losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())



#ezek lesznek a training kimenő értékei
#A trainTargets lesz a training set-ünk, itt adjon hozzá
#a bboxot és a label-eket, másiknál hasonlóan, csak azok a  test lesz
trainTargets = {
    "class_label" : trainLabels,
    "bounding_box" : trainBBoxes
}

#ezek pedig a tesztnek az értékei
testTargets = {
    "class_label" : testLabels,
    "bounding_box" : testBBoxes
}

#Kezdjük el trenírozni a hálózatunkat
print("[Info] training model...")
H = model.fit(
    trainImages, trainTargets, #Megadjuk, hogy mi legyen az amit le eddz és összeasonlítja
    validation_data = (testImages, testTargets), #Itt az eddig nem látott adatbázison teszteli le, hogy mennyire sikere
    batch_size = Config.BATCH_SIZE,
    epochs = Config.NUM_EPOCHS,
    verbose = 1
)

print("[INFO] saving object detector model ...")
model.save(Config.MODEL_PATH, save_format="h5")

print("[INFO] saving label binarizer...")
f = open(Config.LB_PATH, "wb") #Ez azért kel, hogy a binarizált elemeket később tudjuk emberi olvasásra alkalmasan megmutatni
f.write(pickle.dumps(lb))
f.close()
