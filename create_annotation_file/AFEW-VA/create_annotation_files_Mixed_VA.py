import pickle
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
parser = argparse.ArgumentParser(description='read two annotations files')
parser.add_argument('--aff_wild2_pkl', type=str, default = '/home/mvu/Documents/datasets/mixed/affwild-2/annotations.pkl')
parser.add_argument('--VA_pkl', type=str, default = '/home/mvu/Documents/datasets/mixed/afew-va/annotations.pkl')
parser.add_argument('--save_path', type=str, default='/home/mvu/Documents/datasets/mixed/mixed_VA_annotations.pkl')
args = parser.parse_args()
VA_list = ['valence', 'arousal']
def read_aff_wild2():
	total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
	# training set
	train_data = total_data['VA_Set']['Training_Set']
	expr_data = total_data['EXPR_Set']['Training_Set']
	paths = []
	labels = []
	expr_labels = []
	for video in train_data.keys():
		data = train_data[video]
		if video in expr_data.keys():
			data = pd.merge(data, expr_data[video], how='left', on='path').fillna(value=-2)
		else:
			data['label'] = -2
		labels.append(np.stack([data['valence'], data['arousal']], axis=1))
		expr_labels.append(data['label'].values.astype(np.float32))
		paths.append(data['path'].values)
	paths = np.concatenate(paths, axis=0)
	labels = np.concatenate(labels, axis=0)
	expr_labels = np.concatenate(expr_labels, axis=0)
	train_data = {'label': labels, 'path': paths, 'expr': expr_labels}
	# validation set
	val_data = total_data['VA_Set']['Validation_Set']
	paths = []
	labels = []
	for video in val_data.keys():
		data = val_data[video]
		labels.append(np.stack([data['valence'], data['arousal']], axis=1))
		paths.append(data['path'].values)
	paths = np.concatenate(paths, axis=0)
	labels = np.concatenate(labels, axis=0)
	val_data = {'label':labels, 'path':paths}
	return train_data, val_data
def merge_two_datasets():
	data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
	# downsample x 5 the training set in aff_wild training set
	aff_wild_train_labels = data_aff_wild2['label']
	aff_wild_train_paths = data_aff_wild2['path']
	length = len(aff_wild_train_labels)
	index = [True if i%5 ==0 else False for i in range(length)]
	aff_wild_train_labels = aff_wild_train_labels[index]
	aff_wild_train_paths = aff_wild_train_paths[index]
	aff_wild_train_expr = data_aff_wild2['expr'][index]
	data_aff_wild2 = {'label': aff_wild_train_labels, 'path': aff_wild_train_paths, 'expr': aff_wild_train_expr}
	# downsample x 5 the training set in aff_wild
	data_VA = pickle.load(open(args.VA_pkl, 'rb'))
	data_VA = {**data_VA['Training_Set'], **data_VA['Validation_Set']}
	labels =[]
	paths = []
	for video in data_VA.keys():
		data = data_VA[video]
		labels.append(np.stack([data['valence'], data['arousal']], axis=1))
		paths.append(data['path'])
	paths = np.concatenate(paths, axis=0)
	labels = np.concatenate(labels, axis=0)
	data_VA = {'label':labels, 'path':paths, 'expr': np.full((len(paths)), -2)}
	data_merged = {'label': np.concatenate((data_aff_wild2['label'], data_VA['label']), axis=0),
				   'expr': np.concatenate((data_aff_wild2['expr'], data_VA['expr']), axis=0),
	                'path': list(data_aff_wild2['path']) + list(data_VA['path'])}
	print("Aff-wild2 :{}".format(len(data_aff_wild2['label'])))
	print("AFEW_VA:{}".format(len(data_VA['label'])))
	return {'Training_Set': data_merged, 'Validation_Set': data_aff_wild2_val}

def plot_distribution(data):
	all_samples = data['label']
	plt.hist2d(all_samples[:, 0] , all_samples[:, 1] , bins=(20, 20), cmap=plt.cm.jet)
	plt.xlabel("Valence")
	plt.ylabel('Arousal')
	plt.colorbar()
	plt.show()
if __name__== '__main__':
	data_file = merge_two_datasets()
	pickle.dump(data_file, open(args.save_path, 'wb'))
	plot_distribution(data_file['Training_Set'])