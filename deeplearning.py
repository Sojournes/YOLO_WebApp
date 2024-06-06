import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import math
import re


# LOAD YOLO MODEL
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def get_detections(img,net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.1:
            class_score = row[5] # probability score of license plate
            if class_score > 0.1:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.1,0.1)).flatten()
    
    return boxes_np, confidences_np, index

def extract_text(image, bbox):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return '', 0.0
    else:
        # Resize the ROI to expand it along the x-axis to 150%
        new_w = int(w * 2)
        new_h = int(h*1.2)
        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Use EasyOCR to extract text from the resized ROI
        results = reader.readtext(resized_roi, detail=1)
        
        if not results:
            return '', 0.0
        
        # Extract text and confidence scores
        text_list = []
        confidence_list = []
        
        for (bbox, text, confidence) in results:
            text_list.append(text)
            confidence_list.append(confidence)

        if not text_list:
            return "",0
        
        # Join the text parts
        text = ' '.join(text_list).strip()
        
        # Calculate the average confidence score
        avg_confidence = sum(confidence_list) / len(confidence_list)
        
        print(text, avg_confidence, "number")
        return text, avg_confidence


def drawings(image,boxes_np,confidences_np,index):
    # drawings
    if not boxes_np:
        return image,"",0
    for ind in index:
        x,y,w,h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text, correct_factor = extract_text(image,boxes_np[ind])
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)

        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),1)
        
    return image,license_text,correct_factor


# predictions
def yolo_predictions(img, net):
    ## step-1: get detections
    input_image, detections = get_detections(img, net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img,text_result,correct_factor = drawings(img, boxes_np, confidences_np, index)
    return result_img,text_result,correct_factor


def rotate_y(img, y):
    height, width = img.shape[:2]

    # Calculate the new size to avoid cropping
    diagonal = int(math.sqrt(width ** 2 + height ** 2))
    canvas_width = diagonal
    canvas_height = diagonal

    # Create a blank canvas with a larger size
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Calculate the position to place the original image in the center of the canvas
    x_offset = (canvas_width - width) // 2
    y_offset = (canvas_height - height) // 2
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = img

    # Update the projection matrices based on the new canvas size
    proj2dto3d = np.array([[1, 0, -canvas_width / 2],
                           [0, 1, -canvas_height / 2],
                           [0, 0, 0],
                           [0, 0, 1]], np.float32)

    ry = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    trans = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 100], 
                      [0, 0, 0, 1]], np.float32)

    proj3dto2d = np.array([[100, 0, canvas_width / 2, 0],
                           [0, 100, canvas_height / 2, 0],
                           [0, 0, 1, 0]], np.float32)

    ay = float(y * (math.pi / 180))

    ry[0, 0] = math.cos(ay)
    ry[0, 2] = -math.sin(ay)
    ry[2, 0] = math.sin(ay)
    ry[2, 2] = math.cos(ay)

    final = proj3dto2d.dot(trans.dot(ry.dot(proj2dto3d)))

    abs_cos = abs(math.cos(ay))
    abs_sin = abs(math.sin(ay))

    bound_w = int(canvas_height * abs_sin + canvas_width * abs_cos)
    bound_h = int(canvas_height * abs_cos + canvas_width * abs_sin)

    dst = cv2.warpPerspective(canvas, final, (bound_w, bound_h), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (0, 0, 0))
    return dst

def process_images_with_yolo(img, net, angles):
    images = [rotate_y(img, angle) for angle in angles]
    images.append(img)  # Original image

    best_image = None
    highest_correct_factor = 0
    correct_text = None

    for rotated_img in images:
        result_img,text_result, correct_factor = yolo_predictions(rotated_img, net)
        text_result = text_result.replace(" ","")
        match = re.search(r'\b\d{6}\b', text_result)
        if match and correct_factor > highest_correct_factor:
            highest_correct_factor = correct_factor
            best_image = result_img
            correct_text = match.group()

    return best_image,correct_text

angles = [-8,-4,4,8,12]


def object_detection(path,filename):
    # read image
    image = cv2.imread(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    result_img, text_list = process_images_with_yolo(image,net,angles)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    return [text_list]



# def OCR(path,filename):
#     img = np.array(load_img(path))
#     cods = object_detection(path,filename)
#     xmin ,xmax,ymin,ymax = cods[0]
#     roi = img[ymin:ymax,xmin:xmax]
#     roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY)
#     magic_color = apply_brightness_contrast(gray,brightness=40,contrast=70)
#     cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
    
    
    
#     print(text)
#     save_text(filename,text)
#     return text


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf