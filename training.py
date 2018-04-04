import os

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
if split_val>0.:
    train_valid_split = int(split_val * len(train_imgs))
    np.random.shuffle(train_imgs)
    valid_imgs = train_imgs[train_valid_split:]
    train_imgs = train_imgs[:train_valid_split]
else:
    valid_imgs = []

# costruisco il modello
yolo = YOLO2(labels=configObj.classes,
             backend_weights=configObj.backend_weights)

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

# se devo "freezare" i feature extractor, lo faccio
if configObj.do_freeze_layers:
    yolo.feature_extractor.feature_extractor.trainable = False
    # NOTA: eventualmente posso fermare il "freeze" ad un certo layer
    # for l in yolo.feature_extractor.feature_extractor.layers:
    #     if l.name == 'conv_22':
    #         break
    #     l.trainable = False

yolo.model.summary()

yolo.train(train_imgs=train_imgs,
           valid_imgs=valid_imgs,
           nb_epoch=configObj.epochs,
           learning_rate=configObj.base_lr,
           batch_size=configObj.batch_size,
           log_path=configObj.log_path,
           checkpoint_weights_name=configObj.chkpnt_weights_path,
           result_weights_name=configObj.trained_weights_path,
           patience=configObj.patience)
