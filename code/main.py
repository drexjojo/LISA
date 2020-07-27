import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from model import *
from data_loaders import *
from get_embeddings import *
from train import *

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

def get_vocab(data):
	"""
	Gets the unique words for the concerned dataset
	"""
	all_words = []
	for line in data:
		words = line.split()
		all_words += words
	all_words = set(all_words)
	return all_words
		
def load_data(config):

	print("[INFO] -> Loading Preprocessed Data ...")
	model_data_french    = torch.load("../data/data_fr_"+DOMAIN+".pt")
	model_data_english   = torch.load("../data/data_en_"+DOMAIN+".pt")
	model_data_german    = torch.load("../data/data_de_"+DOMAIN+".pt")
	model_data_japanese  = torch.load("../data/data_jp_"+DOMAIN+".pt")
	# model_data_telugu    = torch.load("../data/data_te_"+DOMAIN+".pt")
	print("[INFO] -> Done.")

	print("[INFO] -> Loading Vocabulary ...")
	en_vocab = get_vocab(model_data_english.train_data+model_data_english.test_data)
	fr_vocab = get_vocab(model_data_french.train_data+model_data_french.test_data)
	de_vocab = get_vocab(model_data_german.train_data+model_data_german.test_data)
	jp_vocab = get_vocab(model_data_japanese.train_data+model_data_japanese.test_data)
	# te_vocab = get_vocab(model_data_telugu.train_data+model_data_telugu.test_data)
	print("[INFO] -> Done.")
	
	print("[INFO] -> Loading MUSE Embeddings ...")
	embeddings_model = Emb_Model()
	embeddings_model.get_MUSE_embeddings(en_vocab, fr_vocab, de_vocab, jp_vocab)
	print("The length of the embedding dictionary %d", len(embeddings_model.embeddings))
	print("The length of the word2index dictionary %d", len(embeddings_model.word2index))
	print("The length of the index2word dictionary %d", len(embeddings_model.index2word))
	print("[INFO] -> Done.")

	train_data  = model_data_english.train_data \
				+ model_data_french.train_data \
				+ model_data_german.train_data \
				+ model_data_japanese.train_data

	train_targets = model_data_english.train_targets \
				  + model_data_french.train_targets  \
				  + model_data_german.train_targets  \
				  + model_data_japanese.train_targets

	lang_identifier  = [LANG_DICT["eng"] for i in range(len(model_data_english.train_data))]
	lang_identifier += [LANG_DICT["fra"] for i in range(len(model_data_french.train_data))] 
	lang_identifier += [LANG_DICT["ger"] for i in range(len(model_data_german.train_data))]
	lang_identifier += [LANG_DICT["jap"] for i in range(len(model_data_japanese.train_data))]
	
	train_dset = Driver_Data(
		data = train_data, 
		targets = train_targets,
		lang_identifier = lang_identifier,
		word2index = embeddings_model.word2index)
	
	#For single Lang --------------------------------------

	# lang_identifier = [LANG_DICT["tel"] for i in range(len(model_data_telegu.train_data))]
	# train_dset = Driver_Data(
	# 	data    = model_data_telegu.train_data,
	# 	targets = model_data_telegu.train_targets,
	# 	lang_identifier = lang_identifier,
	# 	word2index = word2index,)

	#-----------------------------------------------------


	train_loader = DataLoader(train_dset, batch_size = config.BATCH_SIZE,shuffle = True, num_workers = 10,collate_fn=pack_collate_fn)

	valid_dset = Driver_Data(
		data = model_data_german.test_data,
		targets = model_data_german.test_targets,
		word2index = embeddings_model.word2index,
		lang_identifier = [LANG_DICT["ger"] for i in range(len(model_data_german.test_data))])

	valid_loader = DataLoader(valid_dset, batch_size = config.BATCH_SIZE, shuffle = False, num_workers = 10,collate_fn=pack_collate_fn)

	return embeddings_model, train_loader, valid_loader

def main():

	wandb.init(project="lisa",config=hyperparameter_defaults)
	config = wandb.config         
	torch.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("[INFO] -> Using Device : ",device)

	embeddings_model, train_loader, valid_loader = load_data(config)

	model = LISA(embeddings=embeddings_model.embeddings,config=config).to(device)
	wandb.watch(model, log="all")
	
	train(model, train_loader, valid_loader, device, config)

if __name__ == '__main__':
	main()
