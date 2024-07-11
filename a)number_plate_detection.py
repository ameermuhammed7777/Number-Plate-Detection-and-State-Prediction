import cv2
import pytesseract

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
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

def main(image_path, cascade_path):
    image = cv2.imread(image_path)
    
    preprocessed_image = preprocess_image(image)
    
    plates = detect_number_plate(image, cascade_path)
    
    for (x, y, w, h) in plates:

        plate_roii = image[y:y+h, x:x+w]
        plate_roi=preprocessed_image[y:y+h,x:x+w]
        
        text = extract_text_from_plate(plate_roi)
        print("Detected Number Plate Text:", text)

        if 'HR' in text[:3]:
            print('Hariyana Registration')
        elif 'GJ' in text[:3]:
            print('Gujarath Registration')   
        elif 'KL' in text[:3]:
            print('Kerala Registration') 
        elif 'AP' in text[:3]:
            print('Andrapradesh Resistration')
        elif 'KA' in text[:3]:
            print('Karnataka Registration')
        elif 'AN' in text[:3]:
            print('Andaman and Nicobar Islands')
        elif 'AR' in text[:3]:
            print('Arunachal Pradesh')   
        elif 'AS' in text[:3]:
            print('Assam') 
        elif 'BR' in text[:3]:
            print('Bihar')
        elif 'CG' in text[:3]:
            print('Chhattisgarh')
        elif 'CH' in text[:3]:
            print('Chandigarh')
        elif 'DL' in text[:3]:
            print('Delhi')   
        elif 'GA' in text[:3]:
            print('Goa') 
        elif 'HP' in text[:3]:
            print('Himachal Pradesh')
        elif 'JH' in text[:3]:
            print('Jharkhand')
        elif 'JK' in text[:3]:
            print('Jammu and Kashmir')
        elif 'LD' in text[:3]:
            print('Lakshadweep')
        elif 'MH' in text[:3]:
            print('Maharashtra')   
        elif 'ML' in text[:3]:
            print('Meghalaya') 
        elif 'MN' in text[:3]:
            print('Manipur')
        elif 'MP' in text[:3]:
            print('Madhya Pradesh')
        elif 'MZ' in text[:3]:
            print('Mizoram')
        elif 'NL' in text[:3]:
            print('Nagaland')   
        elif 'OD' in text[:3]:
            print('Odisha') 
        elif 'PB' in text[:3]:
            print('Punjab')
        elif 'PY' in text[:3]:
            print('Puducherry')
        elif 'RJ' in text[:3]:
            print('Rajasthan')
        elif 'SK' in text[:3]:
            print('Sikkim')   
        elif 'TN' in text[:3]:
            print('Tamil Nadu') 
        elif 'TR' in text[:3]:
            print('Tripura')
        elif 'TS' in text[:3]:
            print('Telangana')
        elif 'UK' in text[:3]:
            print('Uttarakhand')
        elif 'UP' in text[:3]:
            print('Uttar Pradesh')
        elif 'WB' in text[:3]:
            print('West Bengal')
        
        cv2.imshow('Orginal image',image)
        cv2.imshow("Detected Plate", plate_roii)
        cv2.imshow("Thresholded Plate", plate_roi)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

main('/home/ameer/Desktop/Number_Plate_Detection_and_State_Prediction/test/1.jpeg', 
     '/home/ameer/Desktop/Number_Plate_Detection_and_State_Prediction/haarcascade_russian_plate_number.xml')
