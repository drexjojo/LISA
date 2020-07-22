import torch
import unicodedata
from constants import *
from gensim.models import FastText

FT_FOLDER = "../FastText_embs/"
def print_stats(model_data):
	print("\n\n\n NAME : ",model_data.name)
	print("Training data :")
	print("Number of datapoints : ",len(model_data.train_data))
	print("Example : ",model_data.train_data[1])
	print()
	print("Training targets :")
	print("Number of datapoints : ",len(model_data.train_targets))
	print("Example : ",model_data.train_targets[1])
	print()
	print("Test data :")
	print("Number of datapoints : ",len(model_data.test_data))
	print("Example : ",model_data.test_data[1])
	print()
	print("Test targets :")
	print("Number of datapoints : ",len(model_data.test_targets))
	print("Example : ",model_data.test_targets[1])
	print()
	print("Vocab size : ",len(model_data.vocab))
	print()
	pos = 0
	neg = 0
	for tar in model_data.train_targets:
		if tar == 1:
			pos += 1
		elif tar == 0:
			neg += 1
		# else:
		# 	print("error")
		# 	exit(0)

	print("No of positive train examples : ",pos)
	print("No of negitive train examples : ",neg)
	pos = 0
	neg = 0
	for tar in model_data.test_targets:
		if tar == 1:
			pos += 1
		elif tar == 0:
			neg += 1
		# else:
		# 	print("error")
		# 	exit(0)
	
	print("No of positive test examples : ",pos)
	print("No of negitive test examples : ",neg)

class Data_Model:
	def __init__(self, name):

		self.name = name
		self.train_data = []
		self.train_targets = []
		self.test_data = []
		self.test_targets = []

def main():
	
	print("[INFO] -> Loading Preprocessed Data ...")
	model_data_books    	 = torch.load("../data/data_te_books.pt")
	model_data_dvd           = torch.load("../data/data_te_dvd.pt")
	model_data_music         = torch.load("../data/data_te_mukku.pt")	
	model_data_unlabelled    = torch.load("../data/data_te_product.pt")	
	print("[INFO] -> Done.")

	file_name = "telugu_300.txt"

	lines = []
	data = model_data_books.train_data + model_data_books.test_data + model_data_dvd.train_data +model_data_dvd.test_data
	data += model_data_music.train_data + model_data_music.test_data + model_data_unlabelled.train_data
	
	for line in data:
		lines.append(line.split())

	ft_model = FastText(size=300, window=5, min_count=2,workers=10,sg=1)
	ft_model.build_vocab(sentences=lines)
	print("[INFO]-> Training FT embeddings")
	ft_model.train(sentences=lines, total_examples=len(lines), epochs=15)
	# ft_model.save(FT_FOLDER+file_name)
	ft_model.wv.save_word2vec_format(FT_FOLDER+file_name)
	# print_stats(model_data_french)
	print("[INFO] -> Done.")

if __name__ == '__main__':
	main()