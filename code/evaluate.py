import torch
from torch.utils.data import Dataset, DataLoader
from model import *
from data_loaders import *
from get_embeddings import *
from constants import *

def main():

	print("[INFO] -> Loading Preprocessed Data ...")
	model_data_german = torch.load("../data/data_de_"+DOMAIN+".pt")
	print("[INFO] -> Done.")

	# print("[INFO] -> Loading Vocabulary ...")
	# de_vocab = get_vocab(model_data_german.train_data+model_data_german.test_data)
	# print("[INFO] -> Done.")

	# print("[INFO] -> Loading MUSE Embeddings ...")
	# embeddings_model = Emb_Model()
	# embeddings_model.get_MUSE_embeddings(de_vocab)
	# print("The length of the embedding dictionary %d", len(embeddings_model.embeddings))
	# print("The length of the word2index dictionary %d", len(embeddings_model.word2index))
	# print("The length of the index2word dictionary %d", len(embeddings_model.index2word))
	# print("[INFO] -> Done.")

	trained_dict = torch.load(MODEL_PREFIX+MODEL_FILE)
	print(trained_dict["acc"])
	print(trained_dict["run_ID"])
	exit(0)
	# config = trained_dict[""]
	# valid_dset = Driver_Data(
	# 	data = model_data_german.test_data,
	# 	targets = model_data_german.test_targets,
	# 	word2index = embeddings_model.word2index,
	# 	lang_identifier = [LANG_DICT["ger"] for i in range(len(model_data_german.test_data))])

	# valid_loader = DataLoader(valid_dset, batch_size = config.BATCH_SIZE, shuffle = False, num_workers = 10,collate_fn=pack_collate_fn)

	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# print("[INFO] -> Using Device : ",device)

	# model = LSTM_sent_disc(embeddings=embeddings_model.embeddings,config=config).to(device)
	# print(trained_model["acc"])


if __name__=="__main__":
    main()