import matplotlib as mpl
from keras import backend as K
from keras.engine.topology import Container
from keras.layers import Conv2D

mpl.use('Agg')

from model.frontend import YOLO2

from config import Config
import os
import cv2


def crea_dizionario_layer_ricorsivo(layer, dict):
    if isinstance(layer, Container):
        for l in layer.layers:
            crea_dizionario_layer_ricorsivo(l, dict)
    else:
        dict[layer.name] = layer


# leggo la configurazione
config = Config('configYOLO.json')

# se non ci sono pesi specifici, uso i pesi base e le classi base (COCO)
wpath = config.base_weights_path
wname = "BASE"
classes = ['person',
           'bicycle',
           'car',
           'motorcycle',
           'airplane',
           'bus',
           'train',
           'truck',
           'boat',
           'traffic light',
           'fire hydrant',
           'stop sign',
           'parking meter',
           'bench',
           'bird',
           'cat',
           'dog',
           'horse',
           'sheep',
           'cow',
           'elephant',
           'bear',
           'zebra',
           'giraffe',
           'backpack',
           'umbrella',
           'handbag',
           'tie',
           'suitcase',
           'frisbee',
           'skis',
           'snowboard',
           'sports ball',
           'kite',
           'baseball bat',
           'baseball glove',
           'skateboard',
           'surfboard',
           'tennis racket',
           'bottle',
           'wine glass',
           'cup',
           'fork',
           'knife',
           'spoon',
           'bowl',
           'banana',
           'apple',
           'sandwich',
           'orange',
           'broccoli',
           'carrot',
           'hot dog',
           'pizza',
           'donut',
           'cake',
           'chair',
           'couch',
           'potted plant',
           'bed',
           'dining table',
           'toilet',
           'tv',
           'laptop',
           'mouse',
           'remote',
           'keyboard',
           'cell phone',
           'microwave',
           'oven',
           'toaster',
           'sink',
           'refrigerator',
           'book',
           'clock',
           'vase',
           'scissors',
           'teddy bear',
           'hair drier',
           'toothbrush']

# se invece ci sono pesi specifici, uso questi pesi e le classi per cui sono stati trovati
if os.path.isfile(config.trained_weights_path):
    wpath = config.trained_weights_path
    classes = config.classes
    wname = "DEFINITIVI"
elif os.path.isfile(config.pretrained_weights_path):
    wpath = config.pretrained_weights_path
    classes = config.classes
    wname = "PRETRAINED"

# creo il modello
model = YOLO2(backend_weights=config.backend_weights,
              input_size=config.input_size,
              labels=classes)

# carico i pesi
model.load_weights(wpath, by_name=True, skip_mismatch=True)
print("Caricati pesi " + wname)

# creo un dizionario dei layer del modello (ricorsivamente)
layer_dict = dict()
crea_dizionario_layer_ricorsivo(model.feature_extractor.feature_extractor, layer_dict)

# carico le immagini originali e quelle ridimensionate in due array
# ne prendo una alla volta per minimizzare la memoria GPU necessaria
for imgf in os.listdir(config.test_images_path):
    imgfp = os.path.join(config.test_images_path, imgf)
    if os.path.isfile(imgfp):
        orig_image = cv2.resize(cv2.imread(imgfp), (model.input_size, model.input_size))
        cv2.imshow('Immagine', orig_image)

        # mi scorro tutti i layer e lavoro su quelli convoluzionali
        for layer in layer_dict:
            if isinstance(layer_dict[layer], Conv2D):
                # passo l'immagine in input e ottengo i valori di output del layer attuale
                get_layer_output = K.function([model.feature_extractor.feature_extractor.get_input_at(0)],
                                              [layer_dict[layer].output])

                # normalizzo e porto poi nel range 0..255 i valori dell'immagine
                intermediate_output = (get_layer_output([[orig_image]])[0])
                intermediate_output = intermediate_output - intermediate_output.min()
                intermediate_output = intermediate_output / intermediate_output.max()

                # visualizzo le immagini di tutti i filtri del layer attuale
                filter_idx = 0
                while filter_idx < layer_dict[layer].output.shape[3]:

                    filter_img = intermediate_output[0, :, :, filter_idx]
                    cv2.imshow('Filtro', filter_img)

                    # attendo la pressione di un tasto
                    # 'q' esce completamenet
                    # 'n' si sposta sul prossimo layer
                    # '+' avanza di un filtro
                    # '-' indietreggia di un filtro--
                    k = cv2.waitKey()
                    if k == ord('q'):
                        exit(0)
                    if k == ord('n'):
                        break
                    if k == ord('+'):
                        filter_idx += 1
                    if k == ord('-'):
                        filter_idx = max(0, filter_idx - 1)
