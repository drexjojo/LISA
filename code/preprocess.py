# import torch
import unicodedata
# import numpy as np
import string
import re
import nltk
# import spacy
from string import punctuation
from xml.etree.cElementTree import iterparse
# from gensim.models import FastText
# from tqdm import tqdm
from get_embedding import *
from collections import Counter
from constants import *
# from spacy.lang.te import Telugu

lang = 'jp'
DATA_FOLDER = "../data/Amazon_MultiLing_Reviews/"+lang+"/"
# tokenizer = spacy.load(lang)
# tokenizer = Telugu()
class Data_Model:
	def __init__(self, name):

		self.name = name
		self.train_data = []
		self.train_targets = []
		self.test_data = []
		self.test_targets = []

	def clean_data(self, doc):
		# doc = re.sub(r'[^a-z0-9!?\s\.\,]', '' , doc)
		doc = re.sub(r'\.{2,10}', '' , doc)
		doc = re.sub(r'\s+', ' ',doc)
		doc = doc.replace("\n","")
		doc = doc.replace("\\n","")
		return doc
		
	def parse(self,itemfile):
		summaries = []
		rating = []
		reviews = []
		for event, elem in iterparse(itemfile):
			if elem.tag == "summary" :
				summaries.append(elem.text)
			elif elem.tag == "rating" :
				rating.append(elem.text)
			elif elem.tag == "text" :
				reviews.append(elem.text)

		if len(summaries) != len(reviews) :
			print("ERROR in 1")
			exit(0)
		else :
			text = []
			for i,summ in enumerate(summaries) :
				if summ != None and reviews[i] != None :
					text.append(summ + " " + reviews[i])
				elif summ == None and reviews[i] != None :
					text.append(reviews[i])
				elif summ != None and reviews[i] == None:
					text.append(summ)
				else :
					print("ERROR in 2")
					exit(0)

			return text,rating

	#Function to get stop words; Not used now
	def get_stop_words(self):

		useful_words_eng = ['not', 'never', 'less', 'without', 'cannot', 'nobody', 'none',
		'call','after', 'while','beyond', 'several', 'again', 'front',
		'always', 'one', 'two', 'three', 'four', 'five', 'sometime', 
		'another', 'nor', 'on', 'no','more',"nevertheless","more"]

		useful_words_french = ["ne pas","ne","pas","jamais","moins"]

		for i in useful_words:
			if i not in tokenizer.Defaults.stop_words:
				print(i)

		for word in useful_words :
			try:
				tokenizer.Defaults.stop_words.remove(word)
			except KeyError as e:
				pass

		return tokenizer.Defaults.stop_words

	def read_amazon_data(self,topic):

		train_files = [DATA_FOLDER+"music/unlabeled.review"]
		# if topic == "unlabeled":
		# 	train_files = [DATA_FOLDER+"books/unlabeled.review",DATA_FOLDER+"dvd/unlabeled.review",DATA_FOLDER+"music/unlabeled.review"]
		# 	test_files = []
		# else :
		# 	train_files = [DATA_FOLDER+topic+"/train.review"]
		# 	test_files = [DATA_FOLDER+topic+"/test.review"]

		train_data = []
		train_targets = []
		for fname in train_files :
			itemfile = open(fname)
			text, rating = self.parse(itemfile)

			i=0
			for tex in text : #tqdm(text,desc = "Reading training files : "+fname,leave=False) :
				# tokenized_sentence = tokenizer(self.clean_data(self.unicode_to_ascii(tex.lower().strip().strip(punctuation))))

				# if len(tokenized_sentence) != 0:
				# 	tokens = []
				# 	for j in tokenized_sentence:
				# 		if j.lemma_ not in punctuation:
				# 			tokens.append(j.lemma_)

				# 	if float(rating[i]) < 3 :
				# 		label = 0
				# 	else :
				# 		label = 1
				# 	train_data.append(" ".join(tokens))
				# 	train_targets.append(label)
				i+=1
			print(i)
			exit(0)
			itemfile.close()

		self.train_data = train_data
		self.train_targets = train_targets
		
		# test_data = []
		# test_targets = []

		# for fname in test_files:
		# 	itemfile = open(fname)
		# 	text, rating = self.parse(itemfile)

		# 	i=0
		# 	for tex in tqdm(text,desc = "Reading test files : "+fname,leave=False) :
		# 		tokenized_sentence = tokenizer(self.clean_data(self.unicode_to_ascii(tex.lower().strip())))
		# 		if len(tokenized_sentence) != 0:
		# 			tokens = []
		# 			for j in tokenized_sentence:
		# 				if j.lemma_ not in punctuation:
		# 					tokens.append(j.lemma_)

		# 			if float(rating[i]) < 3 :
		# 				label = 0
		# 			else :
		# 				label = 1
		# 			test_data.append(" ".join(tokens))
		# 			test_targets.append(label)
		# 		i+=1
		# 	itemfile.close()

		# self.test_data = test_data
		# self.test_targets = test_targets
		print("[INFO]-> Done.")
		
	def read_telugu_data(self):
		all_pairs = []
		with open("../data/telegu_data.txt") as f:
			for line in f.readlines():
				rating = line.split()[0].strip()
				review = " ".join(line.split()[1:]).strip()
				tokenized_sentence = tokenizer(self.clean_data(self.unicode_to_ascii(review.lower().strip())))
				tokens = []
				for j in tokenized_sentence:
					if j.lemma_ not in punctuation:
						tokens.append(j.lemma_)
				if rating == '+1':
					label = 1
				elif rating == '-1':
					label = 0
				elif rating == '0':
					label = 2
				else:
					label = 2

				all_pairs.append([" ".join(tokens),label])
		self.train_data, self.train_targets, self.test_data, self.test_targets = self.split_data(all_pairs)
		
	def read_Sentirama_data(self):
		all_pairs = []
		with open("../data/Sentiraama/Product Reviews/product_neg.txt") as f:
			for line in f.readlines():
				if line.strip() != "__________________________" :
					tokenized_sentence = tokenizer(self.clean_data(self.unicode_to_ascii(line.lower().strip())))
					tokens = []
					for j in tokenized_sentence:
						if j.lemma_ not in punctuation:
							tokens.append(j.lemma_)
					label = 0
					all_pairs.append([" ".join(tokens),label])

		with open("../data/Sentiraama/Product Reviews/product_pos.txt") as f:
			for line in f.readlines():
				if line.strip() != "__________________________" :
					tokenized_sentence = tokenizer(self.clean_data(self.unicode_to_ascii(line.lower().strip())))
					tokens = []
					for j in tokenized_sentence:
						if j.lemma_ not in punctuation:
							tokens.append(j.lemma_)
					label = 1
					all_pairs.append([" ".join(tokens),label])

		self.train_data, self.train_targets, self.test_data, self.test_targets = self.split_data(all_pairs)

	def split_data(self,all_pairs):

		validation_split = 0.2
		random_seed = 42
		dataset_size = len(all_pairs)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		np.random.seed(random_seed)
		np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		
		train_data = []
		valid_data = []
		train_targets = []
		valid_targets = []

		for ind in tqdm(train_indices,desc = " -> Preparing train data") :
			train_data.append(all_pairs[ind][0])
			train_targets.append(all_pairs[ind][1])

		for ind in tqdm(val_indices,desc = " -> Preparing valid data") :
			valid_data.append(all_pairs[ind][0])
			valid_targets.append(all_pairs[ind][1])

		return train_data, train_targets, valid_data, valid_targets

	def unicode_to_ascii(self, s):
		return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def print_stats(model_data):
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

def main():
	
	topic = "unlabeled"
	model_data = Data_Model("Sentiment_data_"+lang)
	model_data.read_amazon_data(topic)
	# model_data.read_Sentirama_data()
	print("Saving file !")
	# torch.save(model_data, "../data/data_"+lang+"_"+topic+".pt")	
	# torch.save(model_data, "../data/data_te_product.pt")
	# model_data = torch.load("../data/model_data_english.pt")
	print_stats(model_data)

if __name__ == '__main__':
	main()