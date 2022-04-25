import os
#Beaállítjuk az alap mappákat, a képeket és a hozzájuk tartozó annotációkat
BASE_PATH = "/home/miguff/ÓE/Sign_Recognition/dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, 'annotations'])


#Beallítjuk a kimeneti értékeket is, hogy hova megy a model, a tesztek stb.
BASE_OUTPUT = "/home/miguff/ÓE/Sign_Recognition/output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, 'lb.pickle'])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


#Állítsuk be az alap paramétereket a tanításhoz
INIT_LR = 1e-4
NUM_EPOCHS = 40
BATCH_SIZE = 32


