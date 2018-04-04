import os

import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Reshape, Conv2D, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from model.backend import Yolo2Feature
from model.preprocessing import BatchGenerator
from model.utils import decode_netout


class YOLO2(object):
    def __init__(self, labels,
                 backend_weights):

        self.input_size = 608

        self.nb_box = 5
        self.anchors = [0.57273, 0.677385,
                        1.87446, 2.06253,
                        3.33843, 5.47434,
                        7.88282, 3.52778,
                        9.77052, 9.16828]

        self.max_box_per_image = 15

        # preparo il layer di estrazione feature
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))
        self.feature_extractor = Yolo2Feature(self.input_size, backend_weights)

        print("Output shape dell'estrattore di feature: ", self.feature_extractor.get_output_shape())
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        self.reshape_model(labels)

    def reshape_model(self, labels):
        self.labels = list(labels)
        self.nb_class = len(self.labels)
        self.class_wt = np.ones(self.nb_class, dtype='float32')

        input_image = Input(shape=(self.input_size, self.input_size, 3))
        features = self.feature_extractor.extract(input_image)

        # creo il layer di object detection
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])
        self.model = Model([input_image, self.true_boxes], output)

        # inizializzo i pesi del layer di detection
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self.grid_h * self.grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self.grid_h * self.grid_w)

        layer.set_weights([new_kernel, new_bias])

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, 5, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        # correggo predizione x ed y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        # correggo predizione w ed h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        # correggo confidenza
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        # correggo probabilità classificazione
        pred_box_class = y_pred[..., 5:]

        # correggo ground truth x ed y
        true_box_xy = y_true[..., 0:2]

        # correggo ground truth w ed h
        true_box_wh = y_true[..., 2:4]

        # correggo confidenza
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        # correggo probabilità classificazione
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        # maschera delle coordinate: è semplicemente la posizione delle box di ground truth (predittori)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        # maschera della confidenza: penalizza i predittori e i box con un basso IOU
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale

        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        # maschera di classificazione: semplicemente la posizione delle box di ground truth (predittori)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale

        # training di warm-up
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_bs),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh + tf.ones_like(true_box_wh) *
                                                                np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2]) *
                                                                no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])

        # finalizzo la funzione loss
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

        return loss

    def load_weights(self, weight_path, by_name=False, skip_mismatch=False):
        self.model.load_weights(weight_path, by_name=by_name, skip_mismatch=skip_mismatch)

    def predict(self, image):
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:, :, ::-1]

        input_image = np.expand_dims(input_image, 0)
        dummy_array = dummy_array = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes = decode_netout(self.nb_class, netout, self.anchors)

        return boxes

    def train(self, train_imgs,  # lista delle immagini di training
              valid_imgs,  # lista delle immagini di validazione
              log_path,
              checkpoint_weights_name,
              result_weights_name,
              patience,
              train_times=2,  # numero di ripetizioni del training set (per piccoli dataset)
              valid_times=2,  # numero di ripetizioni del validation set (per piccoli dataset)
              nb_epoch=100,  # numero di epoche
              learning_rate=1e-3,  # learning rate di training
              batch_size=32,  # grandezza del batch
              warmup_epochs=3,  # numero di epoche iniziali di "warm-up" che consente al modello di prendere famigliarità con il dataset
              object_scale=0.5,
              no_object_scale=1.0,
              coord_scale=1.0,
              class_scale=1.0,
              debug=False):

        self.batch_size = batch_size
        self.warmup_bs = warmup_epochs * (train_times * (len(train_imgs) / batch_size + 1) + valid_times * (len(valid_imgs) / batch_size + 1))

        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale

        self.debug = debug

        if warmup_epochs > 0:
            nb_epoch = warmup_epochs

        # compilo il modello
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        # preparo i generatori di training e di validazione
        generator_config = {
            'IMAGE_H': self.input_size,
            'IMAGE_W': self.input_size,
            'GRID_H': self.grid_h,
            'GRID_W': self.grid_w,
            'BOX': self.nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self.anchors,
            'BATCH_SIZE': self.batch_size,
            'TRUE_BOX_BUFFER': self.max_box_per_image,
        }

        train_batch = BatchGenerator(train_imgs,
                                     generator_config,
                                     jitter=True,
                                     norm=self.feature_extractor.normalize)
        valid_batch = BatchGenerator(valid_imgs,
                                     generator_config,
                                     norm=self.feature_extractor.normalize,
                                     jitter=False)

        # preparo i vari callback di allenamento
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=1e-4,
                                   patience=patience,
                                   mode='min',
                                   verbose=1)
        checkpoint = ModelCheckpoint(checkpoint_weights_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser(log_path),
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)

        # eseguo il processo di training
        self.model.fit_generator(generator=train_batch,
                                 steps_per_epoch=len(train_batch) * train_times,
                                 epochs=nb_epoch,
                                 verbose=1,
                                 validation_data=valid_batch,
                                 validation_steps=len(valid_batch) * valid_times,
                                 callbacks=[early_stop, checkpoint, tensorboard])

        # salvo i pesi risultanti
        self.model.save_weights(result_weights_name)
