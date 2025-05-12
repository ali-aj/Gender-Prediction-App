import streamlit as st
from PIL import Image
from model import load_model, preprocess_image, predict_gender

# Set page configuration
st.set_page_config(
    page_title="Gender Prediction App",
    page_icon="👤",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 4px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("👤 Gender Prediction App")
st.markdown("""
    Upload an image of a person to predict their gender using our AI model.
    The model will analyze the image and provide a prediction along with confidence scores.
""")

# Load the model
@st.cache_resource
def load_gender_model():
    try:
        classifier = load_model()
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

classifier = load_gender_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        if classifier is not None:
            # Preprocess the image (pipeline handles it, but for interface consistency)
            processed_image = preprocess_image(image)
            
            # Get prediction
            gender, confidence = predict_gender(classifier, processed_image)
            
            # Display results
            st.markdown("### Prediction Results")
            st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Gender: {gender}</h3>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers") 