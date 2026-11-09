# Siamese Neural Network Face Recognition

Face verification / authentication with a Siamese neural network (shared CNN, Euclidean distance,
contrastive loss), Viola-Jones face detection (OpenCV Haar cascade) and a Streamlit interface.
Lab Assignment 06, CSE 4192 (Machine Learning Projects with Python).

## How it works
1. **Training (notebook, done once):** AT&T faces (112x92 grayscale) are turned into positive and negative
   pairs. A shared CNN (Conv-Pool-Conv-Flatten-Dense 128) maps each face to a 128-d embedding; the network
   is trained with contrastive loss (margin 1), Adam, batch size 128.
2. **Enrolment:** several webcam photos -> Viola-Jones detection -> crop -> embedding -> stored in SQLite.
3. **Recognition:** detect the face, compute its embedding, compare with every registered user
   (mean of the 3 smallest distances), accept the closest user only if the distance is below the threshold.

## Project structure
    app.py                      Streamlit interface
    utils.py                    data loading, pairs, shared CNN, distance, loss, accuracy
    siamese_nn.py               Siamese model construction, training, saving
    face_detection.py           Viola-Jones detection and cropping
    onboarding.py               user enrolment
    face_recognition_system.py  verification / authentication
    database.py                 SQLite storage of users and embeddings
    crop_calibration.json       crop geometry for detected faces
    models/                     shared_cnn.h5 (used by the app), siamese_nn.h5, threshold.json
    test_cases/                 test-case results

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Retrain
Download the AT&T Database of Faces into `dataset/att_faces/s1 ... s40` and run the notebook
(`pip install -r requirements-train.txt`).

## Results (held-out identities, from the notebook)
- Threshold selected on all pairs of the 5 held-out identities: T = 0.63 (balanced accuracy 0.960, ROC-AUC 0.984).
- The app's default slider value is 0.58 (stricter, chosen after the TC10 unknown-person test failed at 0.63).
- These numbers are optimistic: the same 5 identities were used to pick the threshold and to evaluate it.

## Deployment notes
- Registered users are stored in `registered_users/` and are **not** in the repository (biometric data).
  On Streamlit Community Cloud the file system is temporary, so registrations disappear when the app restarts.
- For a public deployment set `APP_PASSWORD` in the Streamlit secrets; otherwise anyone can register or delete users.
- The model files are about 75 MB each (GitHub warns above 50 MB and rejects files above 100 MB).

## Limitations
Small training set (35 identities); AT&T studio images differ from webcam images; the Haar detector box
varies from frame to frame and the network is sensitive to that; the false-accept rate on webcam images is
higher than on the dataset; this is a classroom demo, not a security system.
