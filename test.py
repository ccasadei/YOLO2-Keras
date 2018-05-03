import matplotlib as mpl

mpl.use('Agg')

from model.frontend import YOLO2

import numpy as np

from matplotlib import pyplot as plt

from config import Config
import os
import cv2

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

# NOTA: eseguo lo stesso procedimento di freeze perchè questo va a modificare l'ordine dei layer e non potrei più caricare i pesi!
# infatti weights = trained_weights + untrained_weights!
if config.do_freeze_layers:
    # se non c'è il nome del layer su cui fermare il freezing, freezzo tutto il backend
    for l in model.feature_extractor.feature_extractor.layers:
        if l.name == config.freeze_layer_stop_name:
            break
        l.trainable = False

# carico i pesi
if os.path.isfile(wpath):
    model.load_weights(wpath, by_name=True, skip_mismatch=True)
    print("Caricati pesi " + wname)

# carico le immagini originali e quelle ridimensionate in due array
# ne prendo una alla volta per minimizzare la memoria GPU necessaria
for imgf in os.listdir(config.test_images_path):
    imgfp = os.path.join(config.test_images_path, imgf)
    if os.path.isfile(imgfp):
        orig_image = cv2.imread(imgfp)

        boxes = model.predict(orig_image)

        # converte l'immagine correttamente in RGB
        orig_image = orig_image[..., ::-1]

        # prepara i colori
        colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()

        # scrivo i box
        plt.imshow(orig_image)

        current_axis = plt.gca()

        if len(boxes) > 0:
            for box in boxes:
                # trasformo le coordinate normalizzate in coordinate assolute
                xmin = int(box.xmin * orig_image.shape[1])
                xmax = int(box.xmax * orig_image.shape[1])
                ymin = int(box.ymin * orig_image.shape[0])
                ymax = int(box.ymax * orig_image.shape[0])
                color = colors[int(box.label) % len(colors)]
                label = '{}: {:.2f}'.format(classes[int(box.label)], box.score)
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        plt.savefig(os.path.join(config.test_result_path, imgf))
        plt.close()
        print("Elaborata immagine '" + imgf + "'")
