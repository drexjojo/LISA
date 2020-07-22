import io
import numpy as np
import torch
from constants import *

class Emb_Model:
    def __init__(self):
        self.embeddings = []
        self.word2index = {}
        self.index2word = {}
        self.embeddings.append(np.array([0. for i in range(300)]+[1.,0.]))
        self.embeddings.append(np.array([0. for i in range(300)]+[0.,1.]))
        self.word2index[PAD_WORD] = len(self.word2index)
        self.word2index[UNK_WORD] = len(self.word2index)

    def load_MUSE(self,file_name,vocab):
        with io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                if word in vocab and word not in self.word2index.keys():
                    vect = vect +' '+"0." + ' '+"0."
                    vect = np.fromstring(vect, sep=' ')
                    self.embeddings.append(vect)
                    self.word2index[word] = len(self.word2index)

    def get_MUSE_embeddings(self,en_vocab=[],fr_vocab=[],de_vocab=[],jp_vocab=[], te_vocab=[]):
        self.load_MUSE("../muse_embs/unsupervised/vectors-en.txt",en_vocab)
        self.load_MUSE("../muse_embs/unsupervised/vectors-fr.txt",fr_vocab)
        self.load_MUSE("../muse_embs/unsupervised/vectors-de.txt",de_vocab)
        self.load_MUSE("../muse_embs/unsupervised/vectors-jp.txt",jp_vocab)
        self.load_MUSE("../muse_embs/unsupervised/vectors-te.txt",te_vocab)

        self.index2word = {v: k for k, v in self.word2index.items()}
        self.embeddings = np.vstack(self.embeddings)
        

def load_MUSE_vec(emb_path,vocab):
    vectors = []
    vectors.append(np.array([0 for i in range(300)]+[1,0]))
    vectors.append(np.array([0 for i in range(300)]+[0,1]))
    word2id = {}
    word2id[PAD_WORD] = len(word2id)
    word2id[UNK_WORD] = len(word2id)
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            if word in vocab :
                vect = vect +' '+"0" + ' '+"0"
                vect = np.fromstring(vect, sep=' ')
                assert word not in word2id, 'word found twice'
                vectors.append(vect)
                word2id[word] = len(word2id)
            
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))

def main():
    print("[INFO] -> Loading Preprocessed Data ...")
    # model_data_french  = torch.load("../data/model_data_fr.pt")
    # model_data_english = torch.load("../data/model_data_en.pt")
    # model_data_german  = torch.load("../data/model_data_de.pt")
    
    print("[INFO] -> Done.")
    emb_model = Emb_Model()
    emb_model.get_MUSE_embeddings(model_data_english.vocab,model_data_french.vocab, model_data_german.vocab)
    print("Saving file !")
    torch.save(emb_model, "../data/emb_model.pt")    
    # get_nn(src_word, tgt_embeddings, tgt_id2word, tgt_embeddings, tgt_id2word, K=5)

if __name__ == "__main__":
    main()

