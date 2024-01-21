import ImageEffects
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import os
from ultralytics import YOLO

image_df = pd.read_csv('OASIS.csv', index_col = False, usecols=['Theme'])
image_df['FileName'] = image_df['Theme'].apply(lambda x: x + '.jpg')

#applying cropping effect to all images
image_df['FileName'].apply(ImageEffects.auto_cropper)

#applying Sabattier effect to all images
image_df['FileName'].apply(ImageEffects.sabattier_effect)

#applying lower blur effect
image_df['FileName'].apply(lambda x: ImageEffects.blur_effect(x, intensity=3, file_to_save=1))

#appliing higher blur effect
image_df['FileName'].apply(lambda x: ImageEffects.blur_effect(x, intensity=6, file_to_save=2))

#applying pixelation effect
image_df['FileName'].apply(ImageEffects.pixelate_effect)

#applying sharpening effect
image_df['FileName'].apply(ImageEffects.sharpen_effect)





