import copy
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence

from model.utils import BoundBox, bbox_iou


# eseguo il parsing delle annotazioni in formato "Pascal VOC"
def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1]) for i in range(int(len(config['ANCHORS']) // 2))]

        # preparo gli "augmenter" utili per limitare l'overfitting (impostato 50% delle volte)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # definiscio una sequenza di augmentation (non posizionali!) da applicare alle immagini
        self.aug_pipe = iaa.Sequential(
            [
                # esegue da 0 a 5 delle prossime augmentation
                iaa.SomeOf((0, 5),
                           [
                               # converte l'immagine nella sua rappresentazione di superpixel (tessellatura)
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),

                               # esegue vari tipi di blur
                               # blur con sigma tra 0 e 3.0
                               # blur con media locale con kernel tra 2 e 7
                               # blur con mediana locale con kernel tra 2 e 7
                               # iaa.OneOf([
                               #     iaa.GaussianBlur((0, 3.0)),
                               #     iaa.AverageBlur(k=(2, 7)),
                               #     iaa.MedianBlur(k=(3, 11)),
                               # ]),

                               # esegue un sharp dell'immagine
                               # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                               # esegue un emboss dell'immagine
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                               # esegue processi di ricerca dei bordi
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),

                               # aggiunge rumore gaussiano
                               # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

                               # rimuove fino al 10% dei pixel in modo random
                               # iaa.OneOf([
                               #     iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               #     # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               # ]),

                               # inverte i canali del colore
                               # iaa.Invert(0.05, per_channel=True),

                               # cambia la luminosità da -10 a +10
                               # iaa.Add((-10, 10), per_channel=0.5),

                               # cambia la luminosità di un 50-150%
                               # iaa.Multiply((0.5, 1.5), per_channel=0.5),

                               # migliora o peggiora il contrasto
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                               # trasforma in scala di grigi
                               # iaa.Grayscale(alpha=(0.0, 1.0))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  # immagini di input
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))  # ground truth box
        y_batch = np.zeros(
            (r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'], 4 + 1 + self.config['CLASS']))  # output della rete desiderata

        for train_instance in self.images[l_bound:r_bound]:
            # applica l'augmentation
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5 * (obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx = self.config['LABELS'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                        box = [center_x, center_y, center_w, center_h]

                        # trova l'anchor box che predice al meglio il box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assegna ground truth x, y, w, h, confidenza e probabilità di classe a y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                        # assegna il true box a b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # NOTA: decommentare per debug
            # disegno l'immagine e i bounding box per una verifica visuale della corretta impostazione dell'augmentation
            # orig_img = img
            # for obj in all_objs:
            #     if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
            #         cv2.rectangle(orig_img[:, :, ::-1], (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 0, 0), 3)
            #         cv2.putText(orig_img[:, :, ::-1], obj['name'],
            #                     (obj['xmin'] + 2, obj['ymin'] + 12),
            #                     0, 1.2e-3 * orig_img.shape[0],
            #                     (0, 255, 0), 2)
            # cv2.imshow("Debug", orig_img)
            # cv2.waitKey(1000)

            # assegna l'immagine di input a x_batch, eventualmente normalizzata
            if self.norm is not None:
                img = self.norm(img)

            x_batch[instance_count] = img
            instance_count += 1

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h, w, c = image.shape

        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            # scala l'immagine
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            # trasla l'imagine
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            # flippa l'immagine
            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        # ridimensiona l'immagine
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:, :, ::-1]

        # corregge posizione e grandezza degli oggetti
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin

        return image, all_objs
