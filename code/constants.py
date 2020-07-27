PAD = 0
UNK = 1
EMBEDDING_DIM = 302
OUTPUT_SIZE = 2
PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
NUM_LANG = 4
SEED = 42
LANG_DICT = {
	'eng' : 0,
	'fra' : 1,
	'ger' : 2,
	'jap' : 3,
	'tel' : 4,
}

hyperparameter_defaults = dict(
	HIDDEN_SIZE_LSTM = 123,
	HIDDEN_SIZE_LANG = 206,
	HIDDEN_SIZE_SENT = 437,
	LSTM_DROPOUT = 0.2461,
	EPOCH = 104,
	BATCH_SIZE = 412,
	LAMBDA = 0.5576
)

MODEL_PREFIX ="../trained_models/" 
MODEL_FILE = "LR_GER_MUSIC.chkpt"
DOMAIN = "music"


