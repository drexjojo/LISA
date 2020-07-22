import torch
import numpy as np
from torch.utils.data import Dataset
from constants import *

class Data_Model:

	def __init__(self, name):

		self.name = name
		self.train_data = []
		self.train_targets = []
		self.test_data = []
		self.test_targets = []
	
class Driver_Data(Dataset):

	def __init__(self,data,targets,lang_identifier,word2index):
		self.data = data
		self.targets = targets
		self.word2index = word2index
		self.lang_identifier = lang_identifier
		if len(self.targets) != len(self.data):
			print("[INFO] -> ERROR in Driver Data !")
			exit(0)

	def __getitem__(self, index):
		input_seq = self.get_sequence(self.data[index],index)
		target_seq = self.get_target_sequence(self.targets[index],index)
		input_seq = np.array(input_seq)
		input_seq = torch.LongTensor(input_seq).view(-1)
		target_seq = np.array(target_seq)
		target_labels = torch.LongTensor(target_seq).view(-1)

		return input_seq, target_labels

	def get_sequence(self, sentence,index):
		if len(sentence.split()) == 0:
			print("[INFO] -> ERROR empty string found !")
			exit(0)
		indices = []
		for word in sentence.split():
			if word in self.word2index.keys():
				indices.append(self.word2index[word])
			else:
				indices.append(self.word2index[UNK_WORD])
		return indices

	def get_target_sequence(self,target,index):
		target_sequence = [0 for i in range(OUTPUT_SIZE)]
		lang_sequence = []
		target_sequence[target] = 1
		lang_sequence.append(self.lang_identifier[index])
		return target_sequence + lang_sequence

	def __len__(self):
		return len(self.data)

def pack_collate_fn(data):
	data.sort(key=lambda x: len(x[0]), reverse=True)
	sequences,targets = zip(*data)

	lengths = [len(seq) for seq in sequences]

	padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
	for i, seq in enumerate(sequences):
		end = lengths[i]
		padded_seqs[i, :end] = seq[:end]

	targets = torch.stack(targets, 0)
	return padded_seqs,torch.LongTensor(lengths), targets