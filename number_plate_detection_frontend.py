import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import pytesseract

def load_image(path_or_url):
    if path_or_url.startswith(('http://', 'https://')):
        try:
            response = requests.get(path_or_url)
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")
            return None
    else:
        try:
            img = Image.open(path_or_url)
            return img
        except Exception as e:
            st.error(f"Error loading image from file: {e}")
            return None

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image.convert('RGB'))
    return open_cv_image[:, :, ::-1].copy()  

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def detect_number_plate(image, cascade_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    return plates

def extract_text_from_plate(plate_roi):
    plate_roi = cv2.resize(plate_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    return text.strip()

st.title("Number Plate Recognition And State Prediction")
st.write("Paste an image link or a local file path below, and the number plate text will be extracted.")

image_path_or_url = st.text_input("Enter image URL or local file path:")
cascade_path = st.text_input("Enter cascade file path for number plate detection:")

if st.button("Show Registration"):
    if image_path_or_url and cascade_path:
        original_image = load_image(image_path_or_url)
        
        if original_image:
            cv_image = pil_to_cv2(original_image)
            
            preprocessed_image = preprocess_image(cv_image)
            
            plates = detect_number_plate(cv_image, cascade_path)
            
            if len(plates) > 0:
                st.image(original_image, caption="Original Image", use_column_width=True)
                
                for (x, y, w, h) in plates:
                    plate_roi = preprocessed_image[y:y+h, x:x+w]
                    
                    text = extract_text_from_plate(plate_roi)
                    st.write(f"Detected Number Plate Text: {text}")

                    if 'HR' in text[:3]:
                        st.write('Haryana Registration')
                    elif 'GJ' in text[:3]:
                        st.write('Gujarat Registration')   
                    elif 'KL' in text[:3]:
                        st.write('Kerala Registration') 
                    elif 'AP' in text[:3]:
                        st.write('Andhra Pradesh Registration')
                    elif 'KA' in text[:3]:
                        st.write('Karnataka Registration')
                    elif 'AN' in text[:3]:
                        st.write('Andaman and Nicobar Islands')
                    elif 'AR' in text[:3]:
                        st.write('Arunachal Pradesh')   
                    elif 'AS' in text[:3]:
                        st.write('Assam') 
                    elif 'BR' in text[:3]:
                        st.write('Bihar')
                    elif 'CG' in text[:3]:
                        st.write('Chhattisgarh')
                    elif 'CH' in text[:3]:
                        st.write('Chandigarh')
                    elif 'DL' in text[:3]:
                        st.write('Delhi')   
                    elif 'GA' in text[:3]:
                        st.write('Goa') 
                    elif 'HP' in text[:3]:
                        st.write('Himachal Pradesh')
                    elif 'JH' in text[:3]:
                        st.write('Jharkhand')
                    elif 'JK' in text[:3]:
                        st.write('Jammu and Kashmir')
                    elif 'LD' in text[:3]:
                        st.write('Lakshadweep')
                    elif 'MH' in text[:3]:
                        st.write('Maharashtra')   
                    elif 'ML' in text[:3]:
                        st.write('Meghalaya') 
                    elif 'MN' in text[:3]:
                        st.write('Manipur')
                    elif 'MP' in text[:3]:
                        st.write('Madhya Pradesh')
                    elif 'MZ' in text[:3]:
                        st.write('Mizoram')
                    elif 'NL' in text[:3]:
                        st.write('Nagaland')   
                    elif 'OD' in text[:3]:
                        st.write('Odisha') 
                    elif 'PB' in text[:3]:
                        st.write('Punjab')
                    elif 'PY' in text[:3]:
                        st.write('Puducherry')
                    elif 'RJ' in text[:3]:
                        st.write('Rajasthan')
                    elif 'SK' in text[:3]:
                        st.write('Sikkim')   
                    elif 'TN' in text[:3]:
                        st.write('Tamil Nadu') 
                    elif 'TR' in text[:3]:
                        st.write('Tripura')
                    elif 'TS' in text[:3]:
                        st.write('Telangana')
                    elif 'UK' in text[:3]:
                        st.write('Uttarakhand')
                    elif 'UP' in text[:3]:
                        st.write('Uttar Pradesh')
                    elif 'WB' in text[:3]:
                        st.write('West Bengal')

                    st.image(cv_image[y:y+h, x:x+w], caption="Detected Plate", use_column_width=True)
                    st.image(plate_roi, caption="Thresholded Plate", use_column_width=True)
            else:
                st.write("No number plates detected.")
    else:
        st.error("Please enter a valid image URL or local file path and cascade file path.")