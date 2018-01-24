import pickle
import os 
import pandas as pd
from PIL import Image
import numpy as np

p_data = open('se_data.pickle','ab')
labels = pd.read_csv('labels.csv')
breed = set(labels['breed'])
n = len(labels)
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

for i in range(n):
	k = {}
	tmp = Image.open('./train/%s.jpg' % labels['id'][i])
	tmp = tmp.resize((64,64), Image.ANTIALIAS)
	tmp = np.array(tmp)
	k[labels['breed'][i]] = tmp
	pickle.dump(k,p_data)
p_data.close()
