import numpy as np
from PIL import Image, ImageFilter
import os
from ultralytics import YOLO

#automatically crops images to focus on the centre of the most defined object
def auto_cropper(image_file):
    #opening image
    image_path = os.path.join('OriginalOASIS',image_file)
    img = Image.open(image_path)
    #loading pre-trained YOLO object-detection model
    model = YOLO("yolov8n.pt")
    #applying YOLO model on an image
    results = model(source=image_path, show=False, save=False)
    #finds the index of the maximum confidence intervals for each detected object
    max_conf_idx = np.argmax(results[0].boxes.conf)
    #finds the location, width and height of each box (centre coordinates, width and height)
    object_coords = list(float(coordinate) for coordinate in results[0].boxes.xywh[max_conf_idx])
    #calculating cropped image coordinates (contains the middle half of the object's location)
    cropped_coords = tuple([object_coords[0] - (object_coords[2] / 4), 
                           object_coords[1] - (object_coords[3] / 4),
                           object_coords[0] + (object_coords[2] / 4),
                           object_coords[1] + (object_coords[3] / 4)])
    cropped_img = img.crop(cropped_coords)
    return cropped_img
    
#applies algorithm to create sabattier effect at pixel level
def sabattier_algorithm(pixel_value):
    #the sabattier effect inverts the brightness of dark values 
    if pixel_value > 127:
        return 255 - pixel_value
    else:
        return pixel_value

#applies sabattier effect at the image level
def sabattier_effect(image_file):
    #opening image
    image_path = os.path.join('OriginalOASIS',image_file)
    img = Image.open(image_path)
    #converting image to grey-scale
    greyscale_img = img.convert('L')
    #converting the image to a numpy array
    greyscale_array = np.array(greyscale_img)
    #applying the sabbatier algorithm to the colours in the image
    edited_array = np.vectorize(sabattier_algorithm)(greyscale_array)
    #converting back to an image
    sabbatier_img = Image.fromarray(edited_array.astype('uint8'))
    return sabbatier_img

#applies a blur effect to images
def blur_effect(image_file,intensity):
    #opening image
    image_path = os.path.join('OriginalOASIS',image_file)
    img = Image.open(image_path)
    #blurring the image
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius = intensity))
    return blurred_img

def pixelate_effect(image_file):
    #opening image
    image_path = os.path.join('OriginalOASIS',image_file)
    img = Image.open(image_path)
    #getting original resolution (pixel dimensions)
    original_res = img.size
    pixelated_img =  img.resize((int(original_res[0]/5),int(original_res[1]/5)), resample = Image.NEAREST)
    pixelated_img = pixelated_img.resize(original_res)
    return pixelated_img





