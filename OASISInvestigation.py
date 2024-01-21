import cv2
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

image_df = pd.read_csv('OASIS.csv', index_col = False, usecols=['Theme','Category','Valence_mean','Valence_SD','Arousal_mean','Arousal_SD'])

def haralick_calculator(image_name, file_theme):
    image_path = os.path.join(file_theme + 'OASIS', image_name.strip() + '.jpg')
    if image_name.strip() + '.jpg' in os.listdir(file_theme + 'OASIS'):    
        image = cv2.imread(image_path)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grey_image = cv2.normalize(grey_image, None, 0, 255, cv2.NORM_MINMAX)
        haralick_features = np.mean(mh.features.haralick(grey_image.astype(int)), axis=0)
        return pd.Series({'Contrast': haralick_features[1], 'Entropy':haralick_features[8], 'Sum Entropy':haralick_features[7], 'Difference Entropy':haralick_features[10]})  
    else:
        return pd.Series({'Contrast':np.nan, 'Entropy':np.nan, 'Sum Entropy':np.nan, 'Difference Entropy':np.nan})

image_df[['Contrast','Entropy','Sum_Entropy','Difference_Entropy']] = image_df['Theme'].apply(lambda x: haralick_calculator(x, file_theme='Original'))
    
model = smf.ols(formula='Arousal_mean ~ Difference_Entropy', data = image_df).fit()
print(model.summary())

mixed_model = smf.mixedlm("Arousal_mean ~ Difference_Entropy", data=image_df, groups=image_df['Category']).fit(method='bfgs')
print(mixed_model.summary())


#when analysing difference entropy, there is significant variation between categories for arousal but little variation for valence


plt.scatter(image_df['Contrast'], image_df['Entropy'], c = image_df['Valence_mean'], cmap='Reds')
plt.xlim([0,2000])
plt.ylabel('Entropy')
plt.xlabel('Contrast')
cbar = plt.colorbar()
cbar.set_label('Valence')
plt.show()

print(image_df.groupby('Category')['Arousal_mean'].mean())


#applies linear regression seperately on each of the different image categories
def linear_regression_calc(dataframe, category, dependent, independent):
    if category in dataframe['Category'].unique():
        values = image_df[image_df['Category'] == category]
        model = smf.ols(formula= dependent + ' ~ ' + independent, data = values).fit()
        print(model.summary())
    else:
        return 'Category does not exist'
    
for category in ['Animal','Object','Person','Scene']:
    linear_regression_calc(image_df, category, 'Arousal_mean', 'Contrast')
        
        
#print(np.std(image_df['Entropy']))

#print(np.argmax(image_df['Valence_mean']))
print(image_df['Category'].unique())


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
