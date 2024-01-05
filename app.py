from Deblurr_Image.pipeline.InferencePipeline import swin_api
import streamlit as st
from PIL import Image



def main():

    st.set_page_config(
        page_title= 'Welcome to the magic of Self Supervised Learning',
        page_icon= '💁',
        layout= 'centered'
    )

    with st.sidebar :
        st.title("Swaraj Bari")
        image = Image.open('./image/Pic.png')
        st.image(image, width=150)
        st.markdown('''
        #### This is a end to end implementation of Chat Pdf application.

        ##### Technology Used:  Streamlit, LangChain, OpenAI's LLM model
        ##### Skills showcased: Python, OOPs, Machine Learning, MLOps, Natural Language Processing, Dockers, Kubernetes, Google Cloud Platform, git, Problem-Solving,Streamlit, MLFlow, DVC
        
        ##### Contact Info: 
        
        - [Whatsapp](+91 8209942039)
        - [Github Profile]()
        - [Github Repo for this Project]()
        - [LinkedIN Profile]()


        ''')

    st.header("Image Recovery with SwinIR Transformer")

    image = st.file_uploader(
        label= 'Upload thy image here',
        type= ['png', 'jpg', 'jpeg'],
        key= 'image',
        label_visibility= 'hidden'
    )

    st.image(
            image= image,
            caption= 'Input Image'
        )

    if image:

        image = swin_api(
            image= image
        )

        st.image(
            image= image,
            caption= 'Processed Image'
        )







main()
