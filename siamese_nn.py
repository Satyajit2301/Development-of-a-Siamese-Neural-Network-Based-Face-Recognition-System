"""
siamese_nn.py
-------------
Construction and training of the Siamese Neural Network.

Both input branches use the SAME shared CNN (shared weights). The output is
the Euclidean distance between the two 128-d feature vectors.
"""
import os
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Lambda

import utils


def build_siamese_model(input_shape=utils.INPUT_SHAPE, use_second_pool=False):
    """
    Build the Siamese network with the Keras Functional API.

    Returns
    -------
    siamese : keras.Model  - inputs [img_top, img_bottom] -> distance (batch, 1)
    shared  : keras.Model  - the shared CNN (image -> 128-d embedding)
    """
    shared = utils.create_shared_network(input_shape, use_second_pool)

    input_top = Input(shape=input_shape, name="input_top")
    input_bottom = Input(shape=input_shape, name="input_bottom")

    # the SAME network object is applied to both inputs -> shared weights
    feat_top = shared(input_top)
    feat_bottom = shared(input_bottom)

    distance = Lambda(utils.euclidean_distance,
                      output_shape=(1,),
                      name="distance")([feat_top, feat_bottom])

    siamese = Model(inputs=[input_top, input_bottom],
                    outputs=distance,
                    name="Siamese_Network")
    return siamese, shared


# =====================================================
# COMPILE, TRAIN, SAVE, LOAD
# =====================================================
EPOCHS = 10
BATCH_SIZE = 128


def compile_model(model, learning_rate=1e-3):
    """Compile with contrastive loss (margin = 1), Adam optimiser and the custom accuracy."""
    model.compile(loss=utils.contrastive_loss,
                  optimizer=keras.optimizers.Adam(learning_rate),
                  metrics=[utils.accuracy])
    return model


def train_model(model, tr_pairs, tr_labels, val_pairs, val_labels,
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2):
    """
    Train the Siamese network on image pairs.

    Returns the Keras History object (loss / accuracy per epoch).
    """
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
                        shuffle=True,
                        verbose=verbose)
    return history


def save_models(siamese, shared, model_dir="models"):
    """
    Save the full Siamese model (siamese_nn.h5) and the shared CNN
    feature extractor (shared_cnn.h5, used for embeddings at run time).
    """
    os.makedirs(model_dir, exist_ok=True)
    siamese_path = os.path.join(model_dir, "siamese_nn.h5")
    shared_path = os.path.join(model_dir, "shared_cnn.h5")
    siamese.save(siamese_path)
    shared.save(shared_path)
    return siamese_path, shared_path


def load_siamese_model(path):
    """Load the full Siamese model (needs the custom distance function)."""
    return keras.models.load_model(
        path, compile=False, safe_mode=False,
        custom_objects={"euclidean_distance": utils.euclidean_distance})


def load_embedding_model(path):
    """Load the shared CNN (image -> 128-d embedding). No custom objects needed."""
    return keras.models.load_model(path, compile=False)


def pair_distances(model, pairs, batch_size=128):
    """Distance score for every image pair (returns a 1-D array)."""
    return model.predict([pairs[:, 0], pairs[:, 1]],
                         batch_size=batch_size, verbose=0).ravel()


if __name__ == "__main__":
    # run as a script:  python siamese_nn.py
    (X_tr, y_tr), (X_te, y_te) = utils.get_data("dataset/att_faces", 35)
    p_tr, l_tr = utils.create_pairs(X_tr, y_tr)
    p_te, l_te = utils.create_pairs(X_te, y_te)
    siamese, shared = build_siamese_model()
    compile_model(siamese)
    train_model(siamese, p_tr, l_tr, p_te, l_te)
    print(save_models(siamese, shared))
