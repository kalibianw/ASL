from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from keras import mixed_precision
from keras.api.keras import models, layers, activations, initializers, optimizers, metrics, losses, callbacks

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import cv2
import os

from copy import deepcopy


class AnnoDataModule:
    def __init__(self, dataset_dir_path, rescaling_ratio=1, img_height=None, img_width=None):
        self.DATASET_DIR_PATH = dataset_dir_path
        if self.DATASET_DIR_PATH is not None:
            print(f"Find {len(os.listdir(self.DATASET_DIR_PATH))} class(es).")
            print(os.listdir(self.DATASET_DIR_PATH))
        self.RESCALING_RATIO = rescaling_ratio
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width

    def img_to_np(self):
        fnames = list()
        imgs = list()
        for label_name in tqdm(os.listdir(self.DATASET_DIR_PATH), desc="img_to_np"):
            for img_name in tqdm(os.listdir(f"{self.DATASET_DIR_PATH}/{label_name}/lit/")):
                fname = f"{self.DATASET_DIR_PATH}/{label_name}/lit/{img_name}"
                img = cv2.imread(filename=fname)
                img = cv2.resize(src=img, dsize=(0, 0), fx=self.RESCALING_RATIO, fy=self.RESCALING_RATIO)
                self.IMG_HEIGHT = img.shape[0]
                self.IMG_WIDTH = img.shape[1]

                imgs.append(img)
                fnames.append(fname)

        print(f"image height: {self.IMG_HEIGHT}\nimage width: {self.IMG_WIDTH}")
        return np.array(fnames), np.array(imgs)

    def label_to_np(self):
        fnames = list()
        chars = list()
        landmarks = list()
        for label_name in tqdm(os.listdir(self.DATASET_DIR_PATH), desc="label_to_np"):
            for json_fname in tqdm(os.listdir(f"{self.DATASET_DIR_PATH}/{label_name}/annotation/")):
                fname = f"{self.DATASET_DIR_PATH}/{label_name}/annotation/{json_fname}"
                pd_json = pd.read_json(path_or_buf=fname)
                raw_landmark = pd_json["Landmarks"].to_numpy()
                landmark = list()
                for mark in raw_landmark:
                    landmark.append([mark[0] * self.RESCALING_RATIO, mark[1] * self.RESCALING_RATIO])

                fnames.append(fname)
                chars.append(label_name)
                landmarks.append(landmark)

        return np.array(fnames), np.array(chars), np.array(landmarks)

    def cls_normalization(self, chars):
        ordinal_enc = OrdinalEncoder()
        ordinal_chars = ordinal_enc.fit_transform(np.expand_dims(chars, axis=-1))

        onehot_enc = OneHotEncoder()
        onehot_chars = onehot_enc.fit_transform(ordinal_chars)
        onehot_chars = onehot_chars.toarray()

        return onehot_chars

    def lndmrk_normalization(self, landmarks):
        if (self.IMG_HEIGHT is None) or (self.IMG_WIDTH is None):
            raise Exception("""
            IMG_HEIGHT or IMG_WIDTH is None.
            If you didn't run img_to_np, please determine the image height and width on the initialization process.
            """)
        if len(landmarks.shape) != 3:
            raise Exception("landmarks should be 3d array")

        landmarks[:, :, 0] /= self.IMG_WIDTH
        landmarks[:, :, 1] /= self.IMG_HEIGHT

        return landmarks


class SegDataModule(AnnoDataModule):
    def label_to_np(self):
        fnames = list()
        chars = list()
        seg_imgs = list()
        for label_name in tqdm(os.listdir(self.DATASET_DIR_PATH), desc="label_to_np"):
            for seg_img_fname in tqdm(os.listdir(f"{self.DATASET_DIR_PATH}/{label_name}/segmentation/")):
                fname = f"{self.DATASET_DIR_PATH}/{label_name}/segmentation/{seg_img_fname}"
                seg_img = cv2.imread(fname)
                seg_img = cv2.resize(src=seg_img, dsize=(0, 0), fx=self.RESCALING_RATIO, fy=self.RESCALING_RATIO)
                fnames.append(fname)
                chars.append(label_name)
                seg_imgs.append(seg_img)

        return np.array(fnames), np.array(chars), np.array(seg_imgs)


class AnnoVisualModule:
    def __init__(self, is_chars_normalized=False, is_lndmrks_normalized=False):
        self.is_chars_normalized = is_chars_normalized
        self.is_lndmrks_normalized = is_lndmrks_normalized

    def draw_point(self, imgs, chars, lndmrks):
        img_height = imgs.shape[1]
        img_width = imgs.shape[2]
        if self.is_chars_normalized:
            chars = np.argmax(chars, axis=1)
        if self.is_lndmrks_normalized:
            lndmrks[:, :, 0] *= img_width
            lndmrks[:, :, 1] *= img_height

        for img, char, lndmrk in zip(imgs, chars, lndmrks):
            cv2.imshow(f"{char}", mat=img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            pts = np.reshape(lndmrk, newshape=(-1, 1, 2))
            pts = np.asarray(pts, dtype=int)
            img_clone = img.copy()
            cv2.polylines(img_clone, pts=pts, isClosed=True, color=(255, 255, 255), thickness=5)
            cv2.imshow(f"{char}", mat=img_clone)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def draw_line(self, img, lndmrk, color):
        lndmrk = np.asarray(lndmrk, dtype=int)
        tmp = deepcopy(lndmrk[17:22])
        lndmrk[17:22] = lndmrk[12:17]
        lndmrk[12:17] = tmp

        for i in range(len(lndmrk)):
            if i % 5 == 3 and i < 23:
                for j in range(0, 3):
                    self._line1(img, i + j, lndmrk, color)
                    self._line2(img, i, lndmrk, color)

        # Wrist
        cv2.line(img, lndmrk[0], lndmrk[1], color, 2)
        cv2.line(img, lndmrk[1], lndmrk[3], color, 2)
        cv2.line(img, lndmrk[1], lndmrk[18], color, 2)

        # Thumb
        cv2.line(img, lndmrk[23], lndmrk[1], color, 2)
        cv2.line(img, lndmrk[23], lndmrk[24], color, 2)
        cv2.line(img, lndmrk[24], lndmrk[25], color, 2)

    def _line1(self, img, num, data, color):
        cv2.line(img, data[num], data[num + 1], color, 2)

    def _line2(self, img, num, data, color):
        cv2.line(img, data[num], data[num + 5], color, 2)

    def plot_confusion_matrix(self, cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names)
            plt.yticks(tick_marks, target_names)

        if labels:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()


class SegVisualModule(AnnoVisualModule):
    def draw_point(self, imgs, chars, seg_imgs):
        if self.is_chars_normalized:
            chars = np.argmax(chars, axis=1)

        for img, char, seg_img in zip(imgs, chars, seg_imgs):
            cv2.imshow(f"{char}", mat=img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            overlay_img = cv2.addWeighted(img, 1, seg_img, 0.4, 0)
            cv2.imshow(f"{char}", mat=overlay_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class AnnoTrainModule:
    def __init__(self, input_shape, batch_size, ckpt_path, model_path, log_dir):
        self.INPUT_SHAPE = input_shape
        self.BATCH_SIZE = batch_size

        self.CKPT_PATH = ckpt_path
        self.MODEL_PATH = model_path
        self.LOG_DIR = log_dir

    def create_model(self, num_conv_blocks):
        input_layer = layers.Input(shape=self.INPUT_SHAPE, name="img")
        rescaling_layer = layers.experimental.preprocessing.Rescaling(scale=1 / 255.0)(input_layer)

        x = rescaling_layer

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer=initializers.he_normal(), name="conv2d_0")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        block_cnt = 1
        for i in range(1, num_conv_blocks + 1):
            num_conv_filters = 2 ** (5 + block_cnt)
            x_ = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                               kernel_initializer=initializers.he_normal(), name=f"conv2d_{i}_1")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_1")(x_)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                              kernel_initializer=initializers.he_normal(), name=f"conv2d_{i}_2")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_2")(x)
            x = layers.Add()([x_, x])
            if i == num_conv_blocks:
                x = layers.AvgPool2D(padding="same", strides=(1, 1), name=f"avg_pool_2d")(x)
                break
            x = layers.MaxPooling2D(padding="same", name=f"max_pool_2d_{i}")(x)

            if i % 2 == 0:
                block_cnt += 1

        x = layers.Flatten()(x)

        x = layers.Dense(1024, activation=activations.selu, kernel_initializer=initializers.he_normal())(x)

        cls_out = layers.Dense(26, activation=activations.softmax, kernel_initializer=initializers.he_normal(), name="cls_out")(x)
        lndmrk_out = layers.Dense(52, activation=None, kernel_initializer=initializers.he_normal(), name="lndmrk_out")(x)

        model = models.Model(input_layer, [cls_out, lndmrk_out])
        model.compile(
            optimizer=optimizers.Adam(),
            loss={
                "cls_out": losses.categorical_crossentropy,
                "lndmrk_out": losses.MSE
            },
            metrics={
                "cls_out": metrics.categorical_accuracy,
                "lndmrk_out": metrics.MAE
            },
            run_eagerly=True
        )

        return model

    def train(self, model: models.Model,
              x_train, y_cls_train, y_lndmrk_train,
              x_valid, y_cls_valid, y_lndmrk_valid,
              CALLBACKS_MONITOR):
        model.fit(
            x={"img": x_train}, y={"cls_out": y_cls_train, "lndmrk_out": y_lndmrk_train},
            batch_size=self.BATCH_SIZE,
            epochs=1000,
            callbacks=[
                callbacks.TensorBoard(
                    log_dir=self.LOG_DIR
                ),
                callbacks.ReduceLROnPlateau(
                    monitor=CALLBACKS_MONITOR,
                    factor=0.5,
                    patience=3,
                    verbose=1,
                    min_lr=1e-8
                ),
                callbacks.ModelCheckpoint(
                    filepath=self.CKPT_PATH,
                    monitor=CALLBACKS_MONITOR,
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=True
                ),
                callbacks.EarlyStopping(
                    monitor=CALLBACKS_MONITOR,
                    min_delta=1e-5,
                    patience=11,
                    verbose=1
                )
            ],
            validation_data=(
                {"img": x_valid}, {"cls_out": y_cls_valid, "lndmrk_out": y_lndmrk_valid}
            )
        )
        model.load_weights(
            filepath=self.CKPT_PATH
        )
        model.save(filepath=self.MODEL_PATH)


class SegTrainModule(AnnoTrainModule):
    def create_model(self, num_conv_blocks):
        mixed_precision.set_global_policy("mixed_float16")
        input_layer = layers.Input(shape=self.INPUT_SHAPE, name="img")

        x = input_layer

        block_cnt = 1
        end_block_layers = list()
        last_enc_conv_filters = int()
        for i in range(1, num_conv_blocks + 1):
            num_conv_filters = 2 ** (4 + block_cnt)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                              kernel_initializer=initializers.he_normal(), name=f"enc_conv2d_{i}_1")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_1")(x)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                              kernel_initializer=initializers.he_normal(), name=f"enc_conv2d_{i}_2")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_2")(x)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                              kernel_initializer=initializers.he_normal(), name=f"enc_conv2d_{i}_3")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_3")(x)
            end_block_layers.append(x)
            x = layers.MaxPooling2D(padding="same", name=f"max_pool_2d_{i}")(x)
            last_enc_conv_filters = num_conv_filters

            if i % 2 == 0:
                block_cnt += 1

        block_cnt = 1
        for i in range(1, num_conv_blocks + 1):
            num_conv_filters = last_enc_conv_filters / (2 ** (block_cnt - 1))
            x = layers.UpSampling2D(name=f"up_sampling_2d_{i}")(x)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.selu,
                              kernel_initializer=initializers.he_normal(), name=f"dec_conv2d_{i}_1")(x)
            x = layers.Concatenate()([x, end_block_layers[num_conv_blocks - i]])
            x = layers.BatchNormalization(name=f"bn_{i + num_conv_blocks}_1")(x)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.selu,
                              kernel_initializer=initializers.he_normal(), name=f"dec_conv2d_{i}_2")(x)
            x = layers.BatchNormalization(name=f"bn_{i + num_conv_blocks}_2")(x)
            x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.selu,
                              kernel_initializer=initializers.he_normal(), name=f"dec_conv2d_{i}_3")(x)
            x = layers.BatchNormalization(name=f"bn_{i + num_conv_blocks}_3")(x)

            if i % 2 == 0:
                block_cnt += 1

        output_layer = layers.Conv2D(filters=3, kernel_size=(1, 1), padding="same", activation=None,
                                     kernel_initializer=initializers.he_normal(), name="seg_out", dtype="float32")(x)

        model = models.Model(input_layer, output_layer)
        model.compile(
            optimizer=optimizers.Adam(),
            loss={
                "seg_out": losses.MSE
            },
            metrics={
                "seg_out": losses.MAE
            },
            run_eagerly=True
        )

        return model

    def train(self, *args, **kwargs):
        model = kwargs["model"]
        x_train = kwargs["x_train"]
        x_valid = kwargs["x_valid"]
        y_seg_train = kwargs["y_seg_train"],
        y_seg_valid = kwargs["y_seg_valid"]
        callbacks_monitor = kwargs["callbacks_monitor"]

        model.fit(
            x={"img": x_train}, y={"seg_out": y_seg_train},
            batch_size=self.BATCH_SIZE,
            epochs=1000,
            callbacks=[
                callbacks.TensorBoard(
                    log_dir=self.LOG_DIR
                ),
                callbacks.ReduceLROnPlateau(
                    monitor=callbacks_monitor,
                    factor=0.5,
                    patience=2,
                    verbose=1,
                    min_delta=1e-2,
                    min_lr=1e-8
                ),
                callbacks.ModelCheckpoint(
                    filepath=self.CKPT_PATH,
                    monitor=callbacks_monitor,
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=True
                ),
                callbacks.EarlyStopping(
                    monitor=callbacks_monitor,
                    min_delta=1e-2,
                    patience=5,
                    verbose=1
                )
            ],
            validation_data=(
                {"img": x_valid}, {"seg_out": y_seg_valid}
            )
        )
        model.load_weights(
            filepath=self.CKPT_PATH
        )
        model.save(filepath=self.MODEL_PATH)
