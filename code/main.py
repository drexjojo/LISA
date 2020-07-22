import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from model import *
from data_loaders import *
from get_embeddings import *
from train import *


SAVE_FILE = "../data/trained_models/low_resource_telegu_dvd.chkpt"
DOMAIN = "dvd"

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
		
def main():

	wandb.init(project="LISA")
	wandb.watch_called = False
	config = wandb.config         
	config.batch_size = BATCH_SIZE
	config.test_batch_size = BATCH_SIZE
	config.epochs = EPOCH
	config.seed = 42
	config.log_interval = 10
	torch.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True

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


	train_loader = DataLoader(train_dset, batch_size = BATCH_SIZE,shuffle = True, num_workers = 10,collate_fn=pack_collate_fn)

	valid_dset = Driver_Data(
		data = model_data_german.test_data,
		targets = model_data_german.test_targets,
		word2index = embeddings_model.word2index,
		lang_identifier = [LANG_DICT["ger"] for i in range(len(model_data_german.test_data))])

	valid_loader = DataLoader(valid_dset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 10,collate_fn=pack_collate_fn)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("[INFO] -> Using Device : ",device)

	model = LSTM_sent_disc(embeddings=embeddings_model.embeddings).to(device)
	wandb.watch(model, log="all")
	

	train(model, train_loader, valid_loader, device)

	# print("[INFO] -> Best model")
	# print("-----------------------")
	# best_model = torch.load(SAVE_FILE)
	# print("  -[EPOCH]     : ",best_model["epoch"])
	# print("  -[ACCURACY]  : ",best_model["acc"])
	# print("  -[PRECISION]  : ",best_model["precision"])
	# print("  -[RECALL]  : ",best_model["recall"])
	# print("  -[f1]  : ",best_model["f1"])

if __name__ == '__main__':
	main()
