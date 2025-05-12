# Gender Prediction Web Application

A professional web application built with Streamlit that predicts gender from facial images using a deep learning model.

## Features

- Clean and modern user interface
- Real-time gender prediction
- Confidence score display
- Support for common image formats (JPG, JPEG, PNG)
- Error handling and user feedback

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your pre-trained model weights in the root directory as `model_weights.pth`

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your web browser
2. Click on "Choose an image..." to upload a photo
3. The application will display the uploaded image and show the prediction results
4. Results include the predicted gender and confidence score

## Technical Details

- Built with Streamlit for the web interface
- Uses PyTorch and ResNet18 for the deep learning model
- Implements proper image preprocessing and model inference
- Includes error handling and user feedback

## Requirements

- Python 3.7+
- See requirements.txt for package dependencies