import torch
import unicodedata
import numpy as np
import string
import re
import nltk
import spacy
from string import punctuation
from xml.etree.cElementTree import iterparse
from sklearn.model_selection import train_test_split
from collections import Counter
# from gensim.models import FastText
from tqdm import tqdm
# from get_embedding import *
from collections import Counter
# from constants import *
from spacy.lang.te import Telugu

lang = 'te'
# DATA_FOLDER = "../data/"+lang+"/"


# tokenizer = spacy.load(lang)
tokenizer = Telugu()

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
        
        def read_Sentirama_data(self):
                all_pairs = []
                with open("../data/Sentiraama/Movie Reviews/neg_review") as f:
                        lines = f.read().split('__________________________')
                        for index, line in enumerate(lines):
                                # print("This is a line", line)
                                if(index == len(lines) - 1):
                                        pass
                                else:
                                        if len(line.split()) != 0 :
                                                tokenized_sentence = tokenizer(self.clean_data(self.unicode_to_ascii(line.lower().strip())))
                                                # tokenized_sentence = line.lower().split()
                                                tokens = []
                                                for j in tokenized_sentence:
                                                        if j.lemma_ not in punctuation:
                                                                tokens.append(j.lemma_)
                                                label = 0
                                                all_pairs.append([" ".join(tokens),label])

                with open("../data/Sentiraama/Movie Reviews/pos_review") as f:
                        lines = f.read().split('__________________________')
                        for line in lines:
                                if len(line.split()) != 0 :
                                        tokenized_sentence = tokenizer(self.clean_data(self.unicode_to_ascii(line.lower().strip())))
                                        tokens = []
                                        for j in tokenized_sentence:
                                                if j.lemma_ not in punctuation:
                                                        tokens.append(j.lemma_)
                                        label = 1
                                        all_pairs.append([" ".join(tokens),label])
                self.train_data, self.train_targets, self.test_data, self.test_targets = self.split_data(all_pairs)
                # print("Count of training positives:", self.train_targets.count("1"))
                # print("Count of training negetives:", self.train_targets.count("0"))
                # print("Count of testing positives:", self.test_targets.count("1"))
                # print("Count of training negs:", self.test_targets.count("0"))
                print(Counter(self.train_targets))
                print(Counter(self.test_targets))
        
        def split_data(self,all_pairs):
        #     validation_split = 0.2
        #     random_seed = 42
        #     dataset_size = len(all_pairs)
        #     indices = list(range(dataset_size))
        #     split = int(np.floor(validation_split * dataset_size))
        #     np.random.seed(random_seed)
        #     np.random.shuffle(indices)
        #     train_indices, val_indices = indices[split:], indices[:split]

        #     train_data = []
        #     valid_data = []
        #     train_targets = []
        #     valid_targets = []

        #     for ind in tqdm(train_indices,desc = " -> Preparing train data") :
        #             train_data.append(all_pairs[ind][0])
        #             train_targets.append(all_pairs[ind][1])

        #     for ind in tqdm(val_indices,desc = " -> Preparing valid data") :
        #             valid_data.append(all_pairs[ind][0])
        #             valid_targets.append(all_pairs[ind][1])
                xs = [a[0] for a in all_pairs]
                ys = [a[1] for a in all_pairs]
                # print(ys)
                train_data, valid_data, train_targets, valid_targets = train_test_split(xs, ys, test_size = 0.20, random_state = 0, stratify=ys)
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
    model_data.read_Sentirama_data()
    print("Saving file !")
    torch.save(model_data, "../data/data_te1_dvd.pt")
    print_stats(model_data)

if __name__ == '__main__':
        main()