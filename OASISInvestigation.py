import cv2
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

image_df = pd.read_csv('OASIS.csv', index_col = False, usecols=['Theme','Category','Valence_mean','Valence_SD','Arousal_mean','Arousal_SD'])

def haralick_calculator(image_name):
    image_path = 'OriginalOASIS/' + image_name.strip() + '.jpg'
    image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.normalize(grey_image, None, 0, 255, cv2.NORM_MINMAX)
    haralick_features = np.mean(mh.features.haralick(grey_image.astype(int)), axis=0)
    return pd.Series({'Contrast': haralick_features[1], 'Entropy':haralick_features[8]})

image_df[['Contrast','Entropy']] = image_df['Theme'].apply(haralick_calculator)
    
model = smf.ols(formula='Valence_mean ~ Contrast + Entropy', data = image_df).fit()
print(model.summary())


plt.scatter(image_df['Contrast'], image_df['Entropy'], c = image_df['Valence_mean'], cmap='Reds')
plt.xlim([0,2000])
plt.ylabel('Entropy')
plt.xlabel('Contrast')
cbar = plt.colorbar()
cbar.set_label('Valence')
plt.show()
"""
image = cv2.imread('OriginalOASIS/Lightning 4.jpg')
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_image = cv2.normalize(grey_image, None, 0, 255, cv2.NORM_MINMAX)
haralick_features = mh.features.haralick(grey_image.astype(int))
haralick_features = np.mean(haralick_features, axis=0)
plt.imshow(grey_image, cmap='gray')
print(haralick_features)
"""

"""
unscaled_valence = image_df['Valence_mean']
min_valence = min(unscaled_valence)
max_valence = max(unscaled_valence)

scaled_valence = (unscaled_valence - min_valence) / (max_valence - min_valence)
"""
#use pandas to set up a dataframe with: image names, vad_scores, haralick_scores
