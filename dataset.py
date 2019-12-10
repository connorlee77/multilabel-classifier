import os 
import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

mapping = {'desert':0, 'mountains':1, 'sea':2, 'sunset':3, 'trees':4}

class MultilabelDataset(Dataset):
	def __init__(self, base_data_path, csv_filename, image_path, N_CLASSES, transforms=None):

		csv_path = os.path.join(base_data_path, csv_filename)
		self.data = pd.read_csv(csv_path)
		
		print('Creating image paths.')
		# First column contains the image paths
		self.image_arr = np.asarray(self.data.iloc[:, 0])
		for i, img_filename in enumerate(tqdm.tqdm(self.image_arr)):
			self.image_arr[i] = os.path.join(base_data_path, image_path, img_filename)

		# Second column is the labels
		print('Reading labels.')
		label_list = np.asarray(self.data['labels'])
		self.label_arr = np.zeros((len(label_list), N_CLASSES), dtype=np.float32)
		for i, labels in enumerate(tqdm.tqdm(label_list)):
			for label in labels.strip().split(','):
				self.label_arr[i,mapping[label]] = 1

		self.transforms = transforms
		
	def __getitem__(self, index):
		image_filepath = self.image_arr[index]
		img = Image.open(image_filepath).convert('RGB')

		if self.transforms is not None:
			img = self.transforms(img)
		else:
			img = np.ascontiguousarray(img)
			img = torch.from_numpy(img)
		
		label = torch.from_numpy(self.label_arr[index])

		return (img, label)


	def __len__(self):
		return len(self.image_arr)

		
if __name__ == '__main__':
	DATA_PATH = r'C:\Users\conno\Desktop\datasets\miml_dataset'
	TRAIN_FILE = r'miml_labels_1.csv'
	TEST_FILE = r'miml_labels_2.csv'
	IMAGE_FOLDER = r'images'
	dataset = MultilabelDataset(DATA_PATH, TEST_FILE, IMAGE_FOLDER, N_CLASSES=5)
	img, label = dataset.__getitem__(0)
	print(label)
	plt.imshow(np.asarray(img))
	plt.show()


