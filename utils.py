"""
utils.py
--------
Helper functions for the Siamese Neural Network face recognition project.

Part 1 : dataset loading and preprocessing
Part 2 : positive / negative pair generation
Part 3 : shared CNN, Euclidean distance, contrastive loss, accuracy
"""
import os
import random
import numpy as np
import cv2

# ---------------- global constants ----------------
IMG_HEIGHT = 112
IMG_WIDTH = 92
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)


# =====================================================
# PART 1 : DATASET LOADING AND PREPROCESSING
# =====================================================
def load_img(path, size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Load one image from disk as a grayscale image and resize it.

    Parameters
    ----------
    path : str    - path of the image file
    size : tuple  - (width, height) expected by cv2.resize

    Returns
    -------
    numpy.ndarray of shape (112, 92), dtype uint8
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.shape != (size[1], size[0]):
        img = cv2.resize(img, size)
    return img


def img_to_array(img):
    """
    Convert a grayscale image into a model-ready array.
    Normalises pixels to [0, 1] and adds the channel dimension.

    Returns
    -------
    numpy.ndarray of shape (112, 92, 1), dtype float32
    """
    arr = img.astype("float32") / 255.0
    return arr.reshape(IMG_HEIGHT, IMG_WIDTH, 1)


def get_person_folders(dataset_dir):
    """Return person folder names (s1, s2, ...) sorted numerically."""
    folders = [d for d in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, d))]
    return sorted(folders, key=lambda x: int("".join(c for c in x if c.isdigit())))


def get_data(dataset_dir, num_train_persons=35):
    """
    Load the AT&T face dataset, preprocess it and split it BY IDENTITY.

    Steps: read image -> grayscale -> resize (112x92) -> normalise (0-1)
           -> reshape (112, 92, 1) -> unique label per person.
    The first `num_train_persons` individuals form the training set and the
    remaining individuals form the testing set (no identity overlap).

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        X : float32 array (N, 112, 92, 1)
        y : int array (N,)  - class label (0 .. 39) of each person
    """
    folders = get_person_folders(dataset_dir)
    X, y = [], []
    for label, folder in enumerate(folders):
        folder_path = os.path.join(dataset_dir, folder)
        files = sorted(os.listdir(folder_path),
                       key=lambda f: int(os.path.splitext(f)[0]))
        for f in files:
            img = load_img(os.path.join(folder_path, f))
            X.append(img_to_array(img))
            y.append(label)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int64")

    train_mask = y < num_train_persons
    return (X[train_mask], y[train_mask]), (X[~train_mask], y[~train_mask])


# =====================================================
# PART 2 : POSITIVE / NEGATIVE PAIR GENERATION
# =====================================================
def create_pairs(X, y, return_ids=False, seed=42):
    """
    Create balanced positive and negative image pairs for a Siamese network.

    For every person (class) and every image index n:
      * positive pair : image n and image n+1 of the SAME person   -> label 1
      * negative pair : image n of this person and image n of a
                        RANDOMLY chosen DIFFERENT person           -> label 0
    The number of images used per person is (minimum images per person) - 1,
    so every person contributes equally and positives == negatives.
    Finally all pairs are shuffled.

    Parameters
    ----------
    X : float32 array (N, 112, 92, 1)  - preprocessed images
    y : int array (N,)                 - identity label of each image
    return_ids : bool                  - also return the identity ids of each pair
    seed : int                         - random seed (reproducible pairs)

    Returns
    -------
    pairs  : float32 array (P, 2, 112, 92, 1)
    labels : float32 array (P,)   - 1 = same person, 0 = different persons
    ids    : int array (P, 2)     - (only if return_ids) identity of both images
    """
    rng = random.Random(seed)
    classes = list(np.unique(y))
    class_idx = {c: np.where(y == c)[0] for c in classes}
    min_images = min(len(v) for v in class_idx.values()) - 1

    pairs, labels, ids = [], [], []
    for c in classes:
        for n in range(min_images):
            # ---- positive pair (same person) ----
            i1, i2 = class_idx[c][n], class_idx[c][n + 1]
            pairs.append([X[i1], X[i2]])
            labels.append(1)
            ids.append([c, c])

            # ---- negative pair (different persons) ----
            neg_c = rng.choice([k for k in classes if k != c])
            j1, j2 = class_idx[c][n], class_idx[neg_c][n]
            pairs.append([X[j1], X[j2]])
            labels.append(0)
            ids.append([c, neg_c])

    pairs = np.array(pairs, dtype="float32")
    labels = np.array(labels, dtype="float32")
    ids = np.array(ids, dtype="int64")

    # ---- shuffle ----
    perm = np.random.RandomState(seed).permutation(len(labels))
    pairs, labels, ids = pairs[perm], labels[perm], ids[perm]

    if return_ids:
        return pairs, labels, ids
    return pairs, labels


# =====================================================
# PART 3 : SHARED CNN, EUCLIDEAN DISTANCE, LOSS, ACCURACY
# =====================================================
import keras
from keras import ops
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

EMBEDDING_DIM = 128


def create_shared_network(input_shape=INPUT_SHAPE, use_second_pool=False):
    """
    Build the CNN feature extractor shared by both Siamese branches.

    Architecture (assignment Task 3):
        Input 112x92x1 -> Conv2D(64, 3x3, ReLU) -> MaxPooling2D
        -> Conv2D(64, 3x3, ReLU) -> [optional MaxPooling2D]
        -> Flatten -> Dense(128, Sigmoid)

    Parameters
    ----------
    input_shape : tuple - (height, width, channels)
    use_second_pool : bool - add a second MaxPooling2D after the 2nd Conv2D

    Returns
    -------
    keras.Sequential model mapping an image to a 128-d feature vector
    """
    model = Sequential(name="shared_cnn")
    model.add(Input(shape=input_shape, name="face_input"))
    model.add(Conv2D(64, (3, 3), activation="relu", name="conv1"))
    model.add(MaxPooling2D(name="pool1"))
    model.add(Conv2D(64, (3, 3), activation="relu", name="conv2"))
    if use_second_pool:
        model.add(MaxPooling2D(name="pool2"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(EMBEDDING_DIM, activation="sigmoid", name="embedding"))
    return model


def euclidean_distance(vectors):
    """
    Euclidean (L2) distance between two batches of feature vectors.

    Parameters
    ----------
    vectors : list [v1, v2], each of shape (batch, 128)

    Returns
    -------
    tensor of shape (batch, 1) - one distance per image pair
    """
    v1, v2 = vectors
    sum_square = ops.sum(ops.square(v1 - v2), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.config.epsilon()))


def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss  L = Y * D^2 + (1 - Y) * max(margin - D, 0)^2

    y_true : 1 for a positive pair (same person), 0 for a negative pair
    y_pred : Euclidean distance D between the two embeddings
    margin : 1.0 (assignment value)
    """
    y_pred = ops.reshape(y_pred, (-1,))
    y_true = ops.cast(ops.reshape(y_true, (-1,)), y_pred.dtype)
    positive_term = y_true * ops.square(y_pred)
    negative_term = (1.0 - y_true) * ops.square(ops.maximum(margin - y_pred, 0.0))
    return ops.mean(positive_term + negative_term)


def accuracy(y_true, y_pred, threshold=0.5):
    """
    Pair-classification accuracy: a pair is predicted 'same person' (1)
    when its distance is below the threshold, otherwise 'different' (0).
    """
    y_pred = ops.reshape(y_pred, (-1,))
    y_true = ops.cast(ops.reshape(y_true, (-1,)), y_pred.dtype)
    predicted = ops.cast(y_pred < threshold, y_pred.dtype)
    return ops.mean(ops.cast(ops.equal(y_true, predicted), "float32"))
