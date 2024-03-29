import streamlit as st
from PIL import Image
import io
#from utils.caption_generator import initialising
from utils.cap_generation import query
from utils.model_handler import ModelHandler
import os
# st.set_page_config(page_title="Image Captioning", page_icon="📝")

st.markdown("# Image Captioning")
st.sidebar.header("Image Captioning is Selected")
st.write(
    """Unleash the power of AI to add context and meaning to your images! With our image caption generation tool, transform ordinary images into captivating stories. Witness how AI effortlessly describes the essence of each image, opening doors to endless possibilities in visual communication. Experience the future of image understanding with ease and precision, only with Streamlit!"""
)
st.title("Image Captioning App")

st.write("Upload an image and let's generate a caption!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])
generate = st.button("Generate Caption")
model_handler = ModelHandler()

def process_image(image, preds):
    # Display the uploaded image
    st.image(image, caption= preds, use_column_width=True)

    # Here you would typically use your image captioning model to generate the caption
    # For the sake of example, let's just display a placeholder caption
    st.write(f"Caption: {preds}")

def upload_image(path):
    a = model_handler.predict_step([path])
    pred = a[0]
    return pred

def main():
    #model, feature_extractor, tokenizer = initialising()
    if generate :
        if uploaded_file is not None:
            # Read the image file as bytes
            image_bytes = uploaded_file.read()
            
            # Convert the image bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            image1 = image
            image.save('temp.jpg')
            pred = upload_image('C:\\Users\\91798\\Downloads\\Img_generation_captioning\\temp.jpg')
            process_image(image1, pred)
            file_path = "temp.jpg"  # Example file path

            # Check if the file exists before deleting
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} deleted successfully.")
            else:
                print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    main()
