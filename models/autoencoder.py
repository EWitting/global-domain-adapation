import numpy as np
import tensorflow as tf
from tensorflow import keras

MISSING_LABEL_VALUE = -1


def combined_loss(target_weight: float = 1.0, decoder_weight: float = 1.0):
    """Return a new loss function that combines autoencoder and classification loss,
     where missing labels (target data) are ignored for classification.
     :return: loss function for a [decoder output, classifier output] format"""

    def loss(y_true, y_pred):
        autoencoder_loss = keras.losses.mean_squared_error(y_true[0], y_pred[0])
        classifier_loss = keras.losses.binary_crossentropy(y_true[1], y_pred[1])

        # Handle missing labels for the classifier head
        missing_label_mask = tf.math.equal(y_true[1], MISSING_LABEL_VALUE)
        masked_proportion = tf.reduce_mean(tf.cast(missing_label_mask, tf.float32))

        masked_classifier_loss = tf.where(missing_label_mask, 0.0, classifier_loss)

        # # Compute the weight factor for autoencoder loss
        # weight_factor = tf.where(missing_label_mask, target_weight, 1.0)
        #
        # # Apply the weight factor to the autoencoder loss
        # weighted_autoencoder_loss = weight_factor * autoencoder_loss

        # combine and adjust based on amount of points
        autoencoder_loss = tf.math.reduce_mean(autoencoder_loss)
        masked_classifier_loss = tf.math.reduce_mean(masked_classifier_loss) / (1-masked_proportion)

        return decoder_weight * autoencoder_loss + masked_classifier_loss

    return loss


class Autoencoder:
    def __init__(self, input_dim: int, encoder=None, decoder=None, classifier=None,
                 target_weight: float = 1.0, decoder_weight: float = 1.0):
        self.input_dim = input_dim
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.target_weight = target_weight
        self.decoder_weight = decoder_weight

        self.classifier_model, self.combined_model = self._build_model()
        self.history_ = None

    def _build_model(self):
        # Encoder
        input_layer = keras.Input(shape=(self.input_dim,))
        if self.encoder is None:
            encoded = keras.layers.Dense(10, activation='relu')(input_layer)
        else:
            encoded = self.encoder(input_layer)

        # Decoder
        if self.decoder is None:
            decoder_output = keras.layers.Dense(self.input_dim, activation='linear')(encoded)
        else:
            decoder_output = self.decoder(encoded)

        # Classifier
        if self.classifier is None:
            hidden = keras.layers.Dense(10, activation='relu')(encoded)
            hidden = keras.layers.Dense(10, activation='relu')(hidden)
            classifier_output = keras.layers.Dense(1, activation='sigmoid')(hidden)
        else:
            classifier_output = self.classifier(encoded)

        classifier_model = keras.Model(input_layer, classifier_output)
        combined_model = keras.Model(inputs=input_layer, outputs=[decoder_output, classifier_output])
        combined_model.compile(optimizer='adam',
                               loss=combined_loss(self.target_weight, self.decoder_weight))
        return classifier_model, combined_model

    def fit(self, xs, ys, xt, **fit_params):
        x_train = np.concatenate([xs, xt])
        y_train = np.concatenate([ys, np.full(len(xt), MISSING_LABEL_VALUE)])
        hist = self.combined_model.fit(x_train, [x_train, y_train], shuffle=True, **fit_params)
        self.history_ = hist.history
        return self

    def predict(self, x, verbose=0):
        return self.classifier_model.predict(x, verbose=verbose)
