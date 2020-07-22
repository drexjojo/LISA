import torch
import wandb
import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from constants import *

def get_performance(preds,targets):
	softmax_output = nn.functional.softmax(preds,dim=1)
	rounded_preds = torch.round(softmax_output)
	targets = targets.detach().cpu().numpy()
	predictions = rounded_preds.detach().cpu().numpy()
	f1 = f1_score(targets,predictions,average="macro")
	precision = precision_score(targets,predictions,average="macro")
	accuracy = accuracy_score(targets,predictions)
	recall = recall_score(targets,predictions,average="macro")
	return accuracy*100, precision, recall, f1

def train_epoch(model, training_data, optimizer, language_loss_func, sentiment_loss_func, device):
	model.train()
	epoch_loss = 0
	epoch_loss_sentiment = 0
	epoch_loss_language = 0
	epoch_acc = 0
	epoch_precision = 0
	epoch_recall = 0
	epoch_f1 = 0  

	for batch in tqdm(training_data, mininterval=2,desc='  - (Training)   ', leave=False):

		sequences,sequence_lengths,targets = batch
		sequences = sequences.to(device)
		sequence_lengths = sequence_lengths.to(device)
		
		optimizer.zero_grad()

		sentiment_targets = torch.FloatTensor(targets[:,:OUTPUT_SIZE].numpy()).to(device)
		language_targets = targets[:,OUTPUT_SIZE:].squeeze(1).to(device)

		# For lang disc
		sentiment_predictions, lang_predictions = model(sequences,sequence_lengths,device)
		language_loss  = language_loss_func(lang_predictions,language_targets)
		sentiment_loss = sentiment_loss_func(sentiment_predictions,sentiment_targets)
		loss = language_loss + sentiment_loss	
		acc,precision, recall, f1 = get_performance(sentiment_predictions, sentiment_targets)

		# For no disc
		# sentiment_predictions = model(sequences,sequence_lengths,device)
		# sentiment_loss = sentiment_loss_func(sentiment_predictions,sentiment_targets)
		# acc,precision, recall, f1 = get_performance(sentiment_predictions, sentiment_targets)
		# loss = sentiment_loss
		
		loss.backward()
		optimizer.step()
		epoch_loss += float(loss.item())
		epoch_loss_sentiment += float(sentiment_loss.item())
		epoch_loss_language += float(language_loss.item())
		epoch_loss += float(loss.item())
		epoch_acc += float(acc)
		epoch_precision += float(precision)
		epoch_recall += float(recall)
		epoch_f1 += float(f1)

	epoch_loss /= len(training_data)
	epoch_loss_sentiment /= len(training_data)
	epoch_loss_language /= len(training_data)
	epoch_acc /= len(training_data)
	epoch_precision /= len(training_data)
	epoch_recall /= len(training_data)
	epoch_f1 /= len(training_data)

	return epoch_loss_sentiment, epoch_loss_language,epoch_acc,epoch_precision,epoch_recall,epoch_f1

def eval_epoch(model, valid_data, optimizer, language_loss_func, sentiment_loss_func, device):
	model.eval()
	epoch_loss = 0
	epoch_loss_sentiment = 0
	epoch_loss_language = 0
	epoch_acc = 0
	epoch_precision = 0
	epoch_recall = 0
	epoch_f1 = 0  

	with torch.no_grad():
		for batch in tqdm(valid_data, mininterval=2,desc='  - (Validating)   ', leave=False):
			sequences,sequence_lengths,targets = batch
			sequences = sequences.to(device)
			sequence_lengths = sequence_lengths.to(device)
			# targets = targets.to(device)
			optimizer.zero_grad()

			sentiment_targets = torch.FloatTensor(targets[:,:OUTPUT_SIZE].numpy()).to(device)
			language_targets = targets[:,OUTPUT_SIZE:].squeeze(1).to(device)
		
			#For lang disc
			sentiment_predictions, lang_predictions = model(sequences,sequence_lengths,device)
			sentiment_loss = sentiment_loss_func(sentiment_predictions,sentiment_targets)
			language_loss  = language_loss_func(lang_predictions,language_targets)
			loss = sentiment_loss + language_loss	
			acc,precision, recall, f1 = get_performance(sentiment_predictions, sentiment_targets)

			#For no disc
			# sentiment_predictions = model(sequences,sequence_lengths,device)
			# sentiment_loss = sentiment_loss_func(sentiment_predictions,sentiment_targets)
			# acc,precision, recall, f1 = get_performance(sentiment_predictions, sentiment_targets)
			# loss = sentiment_loss
			
			epoch_loss += float(loss.item())
			epoch_loss_sentiment += float(sentiment_loss.item())
			epoch_loss_language += float(language_loss.item())
			epoch_loss += float(loss.item())
			epoch_acc += float(acc)
			epoch_precision += float(precision)
			epoch_recall += float(recall)
			epoch_f1 += float(f1)

	epoch_loss /= len(valid_data)
	epoch_loss_sentiment /= len(valid_data)
	epoch_loss_language /= len(valid_data)
	epoch_acc /= len(valid_data)
	epoch_precision /= len(valid_data)
	epoch_recall /= len(valid_data)
	epoch_f1 /= len(valid_data)

	wandb.log({
        "Test Accuracy": epoch_acc,
        "Test Sentiment Loss": epoch_loss_sentiment,
        "Test Language Loss": epoch_loss_language})

	return epoch_loss_sentiment, epoch_loss_language,epoch_acc,epoch_precision,epoch_recall,epoch_f1

def train(model,training_data, validation_data, device):

	language_loss_func  = nn.CrossEntropyLoss().to(device)
	sentiment_loss_func = nn.BCEWithLogitsLoss().to(device)
	optimizer = optim.Adam(model.parameters())

	max_valid_accuracy = 0
	for epoch_i in range(EPOCH):
		print('[ Epoch', epoch_i, ']')

		start = time.time()
		train_loss_sentiment, train_loss_language, train_accu, train_precision, train_recall, train_f1 = train_epoch(model,training_data, optimizer, language_loss_func, sentiment_loss_func,device)
		print('  - (Training)     Loss_sent: {ppl_sent: 8.5f} <-> Loss_lang: {ppl_lang: 8.5f} <-> Accuracy: {accu:3.3f} % <-> Precision: {pres:3.3f} <-> Recall: {rec:3.3f} <-> F1: {fa1:3.3f} <-> '\
		'Time Taken : {elapse:3.3f} min'.format(
		ppl_sent=train_loss_sentiment,ppl_lang=train_loss_language, accu=train_accu, pres=train_precision, rec=train_recall, fa1= train_f1,
		elapse=(time.time()-start)/60))

		start = time.time()
		valid_loss_sentiment, valid_loss_language, valid_accu, valid_precision, valid_recall, valid_f1 = eval_epoch(model,validation_data, optimizer, language_loss_func, sentiment_loss_func,device)
		print('  - (Validating)   Loss_sent: {ppl_sent: 8.5f} <-> Loss_lang: {ppl_lang: 8.5f} <-> Accuracy: {accu:3.3f} % <-> Precision: {pres:3.3f} <-> Recall: {rec:3.3f} <-> F1: {fa1:3.3f} <-> '\
		'Time Taken : {elapse:3.3f} min'.format(
		ppl_sent=valid_loss_sentiment,ppl_lang=valid_loss_language, accu=valid_accu, pres=valid_precision, rec=valid_recall, fa1= valid_f1,
		elapse=(time.time()-start)/60))
		
		if valid_accu >= max_valid_accuracy:
			max_valid_accuracy = valid_accu
		# 	model_state_dict = model.state_dict()
		# 	checkpoint = {'model': model_state_dict,
		# 					'epoch': epoch_i,
		# 					'acc':valid_accu,
		# 					'loss':valid_loss_sentiment,
		# 					'precision':valid_precision,
		# 					'recall':valid_recall,
		# 					'f1':valid_f1 }
		# 	torch.save(checkpoint, SAVE_FILE)
			print('[INFO] -> The checkpoint file has been updated.')
		print(max_valid_accuracy)
