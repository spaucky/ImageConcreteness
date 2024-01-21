from OASISInvestigation import haralick_calculator
import pandas as pd
import cv2
import mahotas as mh
import os

image_df = pd.read_csv('OASIS.csv', index_col = False, usecols=['Theme','Category','Valence_mean','Valence_SD','Arousal_mean','Arousal_SD'])

#calculating original haralick features
image_df[['Contrast','Entropy','Sum_Entropy','Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(image_name = x.strip(), file_theme='Original'))

#calculating cropped haralick features
image_df[['Cropped_Contrast','Cropped_Entropy','Cropped_Sum_Entropy','Cropped_Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(image_name = 'Cropped ' + x.strip(), file_theme='Cropped'))

#calculating sabattier haralick features
image_df[['Sabattier_Contrast','Sabattier_Entropy','Sabattier_Sum_Entropy','Sabattier_Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(image_name = 'Sabattier ' + x.strip(), file_theme='Sabattier'))

#calculating blurred1 haralick features
image_df[['Blurred1_Contrast','Blurred1_Entropy','Blurred1_Sum_Entropy','Blurred1_Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(image_name = 'Blurred ' + x.strip(), file_theme='Blurred1'))

#calculating blurred2 haralick features
image_df[['Blurred2_Contrast','Blurred2_Entropy','Blurred2_Sum_Entropy','Blurred2_Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(image_name = 'Blurred ' + x.strip(), file_theme='Blurred2'))

#calculating pixelated haralick features
image_df[['Pixelated_Contrast','Pixelated_Entropy','Pixelated_Sum_Entropy','Pixelated_Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(image_name = 'Pixelated ' + x.strip(), file_theme='Pixelated'))

#calculating sharpened haralick features
image_df[['Sharpened_Contrast','Sharpened_Entropy','Sharpened_Sum_Entropy','Sharpened_Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(image_name = 'Sharpened ' + x.strip(), file_theme='Sharpened'))

image_df.to_csv('AffectedHaralickFeatures.csv')
