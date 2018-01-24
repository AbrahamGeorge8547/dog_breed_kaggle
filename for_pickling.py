import pandas as pd
import os
import numpy as np
import pickle
from PIL import Image
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labels = pd.read_csv('labels.csv')
labels_test = pd.read_csv('sample_submission.csv')


def img_loader(labels, size,flag):
	out = []
	for i in range(len(labels)):
		tmp = []
		fid=labels['id'][i]
		tmp.append(fid)
		if flag == 0:
			img = Image.open('./train/%s.jpg' % labels['id'][i])
		else:
			img = Image.open('./test/%s.jpg' % labels['id'][i])
		img = img.resize((size, size), Image.ANTIALIAS)
		img_arr = np.array(img)
		tmp.append(img_arr)
		out.append(tmp)
	return out
	
labels =labels.sort_values("id")
train_data = img_loader(labels, 32, 0)
test_data = img_loader(labels_test, 32, 1)
train_data = pd.DataFrame(train_data, columns=['id','image'])
test_data = pd.DataFrame(test_data, columns=['id','image'])
train_data['scaled'] = train_data['image'] / 255
test_data['scaled'] = test_data['image'] / 255
train_data = train_data.drop('image', axis=1)
test_data = test_data.drop('image', axis=1)
train_data = train_data.merge(labels,on ='id')
train_data = train_data.sort_values(by='breed')
train_data =  train_data.drop('breed', axis=1)
labels =labels.sort_values("breed")
labels = labels['breed'].tolist()
label_encoder =LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
one_hot =OneHotEncoder(sparse=False)
integer_encoded =integer_encoded.reshape(len(integer_encoded), -1)
onehot = one_hot.fit_transform(integer_encoded)
label_encoder =LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
one_hot =OneHotEncoder(sparse=False)
integer_encoded =integer_encoded.reshape(len(integer_encoded), -1)
onehot = one_hot.fit_transform(integer_encoded)

with open(r"dog_breed32.pkl","wb") as outputfile:
	pickle.dump([train_data, test_data], outputfile)
outputfile.close()
with open(r"label32.pkl", "wb") as filew:
	pickle.dump(onehot,filew)
filew.close()



 
