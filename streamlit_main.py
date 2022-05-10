import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("mdl_wt.hdf5")
# load file
uploaded_file = st.file_uploader("Choose a image file of "
                                 "1. Butterfly"
                                 " 2. Cat"
                                 " 3. Chicken"
                                 " 4. Cow"
                                 " 5. Dog"
                                 " 6. Elephant"
                                 " 7. Horse"
                                 " 8. Sheep"
                                 " 9. Spider"
                                 " 10. Squirrel", type="jpeg")

map_dict = {0: 'Butterfly',
            1: 'Cat',
            2: 'Chicken',
            3: 'Cow',
            4: 'Dog',
            5: 'Elephant',
            6: 'Horse',
            7: 'Sheep',
            8: 'Spider',
            9: 'Squirrel'}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (224, 224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    predict = st.button("Go For Prediction")
    if predict:
        prediction = model.predict(img_reshape).argmax()
        st.title("The image is {}".format(map_dict[prediction]))
