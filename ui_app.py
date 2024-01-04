import streamlit as st
from tensorflow import keras
import pandas as pd
import numpy as np
from keras.utils import image_utils
from keras import backend as K
from PIL import Image


@st.cache(allow_output_mutation=True)
def prepare_model():
    """
    Preparing a keras model, it runs only once when application is started
    """
    main_model = keras.models.load_model(
        "model\\model.h5")
    main_model.make_predict_function()
    session = K.get_session()
    return main_model, session
    

st.markdown(
    """ <style> .font {
font-size:35px ; font-family: 'Cooper Black'; color: #f5f5f5;} 
</style> """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="font">Counterfeit currency detection</p>', unsafe_allow_html=True
)


def check_image(path):
    input_img = image_utils.load_img(path, target_size=(400, 400))
    arr = np.array(input_img)
    arr = np.expand_dims(arr, axis=0)
    result = model.predict(arr)
    return result

if __name__ == "__main__":
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    model, session = prepare_model()
    if uploaded_file is not None:
        K.set_session(session)
        image = Image.open(uploaded_file)
        st.image(image, width=400)
        result_arr = check_image(uploaded_file)
        counterfeit = round((float(result_arr[0][0])*100.0), 2)
        genuine = round((float(result_arr[0][1])*100.0), 2)
        counterfeit_txt = f'<p style="font-family:sans-serif; color:Red; font-size: 42px;">Counterfeit: {counterfeit}%</p>'
        genuine_txt = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">Genuine: {genuine}%</p>'
        st.markdown(counterfeit_txt, unsafe_allow_html=True)
        st.markdown(genuine_txt, unsafe_allow_html=True)
        st.write()
