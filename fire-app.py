import streamlit as st
import time
import numpy as np
import tensorflow
from skimage.metrics import structural_similarity as ssim
from skimage import io, color, transform
from tensorflow.keras.preprocessing import image
from PIL import Image
import pickle
import time


# Set the page configuration
st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":rocket:",
    layout="wide",  # Use "wide" layout
    initial_sidebar_state="expanded",  # Expand the sidebar by default
)

with st.spinner('Application Loading...'):
    # Load the model using st.cache
    @st.cache(allow_output_mutation=True)
    def load_model():
        return tensorflow.keras.models.load_model("FireDetection_model.h5")

    model = load_model()



# Add a banner image
#banner_image = "https://images.unsplash.com/photo-1700156102664-75e937f92760?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Replace with the path to your banner image
banner_image = "https://images.squarespace-cdn.com/content/v1/5f7a5c42aa3c2d3994e02de4/1602510730363-XYIUILK1EKM88R4K4YSM/forest-banner.jpg?format=2500w"
st.image(banner_image, use_column_width=True)

# Title and description
st.title("Forest Fire Detection App")

st.subheader("This is a Wildfire Detection App. Given an image of a forest, the model predicts where there is prescence of fire in the image or not.")
#st.markdown("Upload an image and receive see where your product fits.")
st.markdown(
    """
    <link rel="stylesheet" type="text/css" href="https://www.example.com/style.css">
    """,
    unsafe_allow_html=True
)
# File uploader
uploaded_image = st.file_uploader("Upload an image of the Forest", type=["jpg", "png", "jpeg"])
image_path2 = "test1.jpeg"









# Image display
if uploaded_image:
    
    # Read the image file
    test_image = Image.open(uploaded_image)
    
     # Read the image file
    #test_image = st.image(uploaded_image)

    # Convert the test image to OpenCV format
    # test_image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
   
    # Display the uploaded image
    st.image(test_image)


    progress_text = "Checking if image shows presence of fire..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()



    # # Calculate image similarity score
    # similarity_score = compare_images(base_image, test_image)
    # #print(similarity_score)
    # print(f"Similarity: {similarity_score}")


    # Check if the similarity score is above a threshold
    # if similarity_score > 0.55:
    #     st.success("Your image is relevant!")

    from PIL import Image
    import numpy as np

    def preprocess_uploaded_image(test_image, img_size=(256, 256)):
        # Assuming test_image is a NumPy array representing an image
        # Convert NumPy array to Pillow image
        # img = Image.fromarray(test_image)

        # # Ensure the image is in RGB mode
        # img = img.convert('RGB')

        # # Resize the image
        # img = img.resize(img_size)

        # # Convert the image to a NumPy array
        # img_array = np.array(img)

        # # Normalize the pixel values to be between 0 and 1
        # img_array = img_array / 255.0

        # # Add an extra dimension to the array to make it (1, height, width, channels)
        # img_array = np.expand_dims(img_array, axis=0)
        img = image.img_to_array(test_image)/255
        img = tensorflow.image.resize(img,(256,256))
        img = tensorflow.expand_dims(img,axis=0)

        return img
            
    # Preprocess the image for the model (resize, normalize, etc.)
    preprocessed_image = preprocess_uploaded_image(test_image)

    # # Convert the preprocessed image to a NumPy array
    # preprocessed_image = np.array(preprocessed_image)

    # Make a prediction using the pickled model
    prediction = model.predict(preprocessed_image)
    # prediction = int(tensorflow.round(model.predict(x=preprocessed_image)).numpy()[0][0])
    print(prediction)
           
    # else:
    #     st.error("Your image is not relevant")
else:
    st.info("Please upload an image to predict.")



# Feedback placeholder
st.subheader("Feedback:")
feedback_text = st.empty()

button_color = "#ff6347"
# Button to trigger feedback generation
if st.button("Generate Feedback"):
    # if similarity_score > 0.55:
    progress_text1 = "Making Prediction..."
    my_bar2 = st.progress(0, text=progress_text1)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar2.progress(percent_complete + 1, text=progress_text1)
    time.sleep(1)
    my_bar2.empty()
        # Read the ima

    # Implement your feedback generation logic here
    if prediction <= 0.55:
        feedback = "There is presence of fire in this image. Sending alert to authorities..."#generate_feedback(uploaded_image)
        #feedback_text.write(feedback)
        st.error(feedback)
    elif prediction > 0.55 :
        feedback = "There is no presence of fire in this image."#generate_feedback(uploaded_image)
        #feedback_text.write(feedback)
        st.success(feedback)
 