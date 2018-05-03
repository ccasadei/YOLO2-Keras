import os
import time

import numpy as np

from config import Config
from model.frontend import YOLO2
from model.preprocessing import parse_annotation

# leggo la configurazione
configObj = Config("configYOLO.json")

# leggo l'elenco di immagini e di label contenuti nel training set
train_imgs, train_labels = parse_annotation(configObj.annotations_path,
                                            configObj.images_path,
                                            configObj.classes)

# suddivido il dataset di due insiemi disgiunti di validation e training
split_val = float(configObj.train_val_split)
if split_val > 0.:
    train_valid_split = int(split_val * len(train_imgs))
    # mescola l'ordine delle righe (casuale, ma ripetibile)
    np.random.seed(19081974)
    np.random.shuffle(train_imgs)
    valid_imgs = train_imgs[train_valid_split:]
    train_imgs = train_imgs[:train_valid_split]
    # resetto il seed random con un numero dipendente dall'istante attuale in millisecondi
    np.random.seed(int(round(time.time() * 1000)) % 2 ** 32)
else:
    valid_imgs = []

# costruisco il modello
yolo = YOLO2(labels=configObj.classes, input_size=configObj.input_size,
             backend_weights=configObj.backend_weights)

yolo.model.summary()

# se devo "freezare" i feature extractor, lo faccio
# NOTA: lo faccio prima di caricare i dati perchè l'ordine dei pesi cambierà (weights = trained_weights + untrained_weights)
if configObj.do_freeze_layers:
    # se non c'è il nome del layer su cui fermare il freezing, freezzo tutto il backend
    conta = 0
    for l in yolo.feature_extractor.feature_extractor.layers:
        if l.name == configObj.freeze_layer_stop_name:
            break
        l.trainable = False
        conta += 1

    print("")
    print("Eseguito freeze di " + str(conta) + " layers")
    print("Nuovo summary dopo FREEZE")
    print("")
    yolo.model.summary()

# carico i pesi pretrained, se ci sono
if os.path.exists(configObj.pretrained_weights_path):
    print("Carico i pesi PRETRAINED")
    yolo.load_weights(configObj.pretrained_weights_path, by_name=True, skip_mismatch=True)
else:
    # carico i pesi base, se ci sono
    if os.path.exists(configObj.base_weights_path):
        print("Carico i pesi BASE")
        yolo.load_weights(configObj.base_weights_path, by_name=True, skip_mismatch=True)
    else:
        print("Pesi PRETRAINED e BASE non trovati")

# for l in yolo.feature_extractor.feature_extractor.layers:
#     l.trainable = False
# yolo.model.save("./h5/pretrained.h5")

yolo.train(train_imgs=train_imgs,
           valid_imgs=valid_imgs,
           config=configObj,
           nb_epoch=configObj.epochs,
           learning_rate=configObj.base_lr,
           batch_size=configObj.batch_size,
           result_weights_name=configObj.trained_weights_path,
           augmentation=configObj.augmentation,
           warmup_epochs=configObj.warmup_epochs)
