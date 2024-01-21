import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

haralick_df = pd.read_csv('AffectedHaralickFeatures.csv')
print(haralick_df.describe().T)











plt.scatter(haralick_df.index, haralick_df['Contrast'],  c='red', label='Original', s=20, alpha=0.5)
plt.scatter(haralick_df.index, haralick_df['Cropped_Contrast'], c='yellow', label='Cropped', s=5, alpha=0.5)
plt.scatter(haralick_df.index, haralick_df['Sabattier_Contrast'], c='blue', label='Sabattier', s=5, alpha=0.5)
plt.scatter(haralick_df.index, haralick_df['Blurred1_Contrast'], c='purple', label='Blurred (radius = 3)', s=5, alpha=0.5)
plt.scatter(haralick_df.index, haralick_df['Blurred2_Contrast'], c='pink', label='Blurred (radius = 6)', s=5, alpha=0.5)
plt.scatter(haralick_df.index, haralick_df['Pixelated_Contrast'], c='green', label='Pixelated', s=5, alpha=0.5)
plt.scatter(haralick_df.index, haralick_df['Sharpened_Contrast'], c='aqua', label='Sharpened', s=5, alpha=0.5)
plt.ylim([0,2000])
plt.legend()
plt.show()


plt.scatter(haralick_df['Difference_Entropy'], haralick_df['Contrast'],  c='red', label='Original', s=20, alpha=0.5)
#plt.scatter(haralick_df['Cropped_Difference_Entropy'], haralick_df['Cropped_Contrast'], c='green', label='Cropped', s=20, alpha=0.5)
#plt.scatter(haralick_df['Sabattier_Difference_Entropy'], haralick_df['Sabattier_Contrast'], c='blue', label='Sabattier', s=5, alpha=0.5)
#plt.scatter(haralick_df['Blurred1_Difference_Entropy'], haralick_df['Blurred1_Contrast'], c='blue', label='Blurred (radius = 3)', s=20, alpha=0.5)
#plt.scatter(haralick_df['Blurred2_Difference_Entropy'], haralick_df['Blurred2_Contrast'], c='pink', label='Blurred (radius = 6)', s=5, alpha=0.5)
#plt.scatter(haralick_df['Pixelated_Difference_Entropy'], haralick_df['Pixelated_Contrast'], c='green', label='Pixelated', s=5, alpha=0.5)
plt.scatter(haralick_df['Sharpened_Difference_Entropy'], haralick_df['Sharpened_Contrast'], c='aqua', label='Sharpened', s=5, alpha=0.5)
#plt.ylim([0,2000])
plt.legend()
plt.show()

