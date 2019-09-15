import os
import pickle
import nltk
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords 
from collections import defaultdict



#email_df = pd.read_csv("enron_all.csv",na_filter= False)
#file_name = list(email_df["Message-ID"].values)
trend_path  = "Enron" 
author_list = os.listdir(trend_path)
#print(len(author_list))
author_dict = {} 
for i, author in enumerate(author_list):
	author_dict[author] = i

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()


def find_trend_index_inside_overall_data(folder_path, file_name_list):
	index_list = []
	author_list = os.listdir(folder_path)
	for author in author_list:
		if author == ".DS_Store":
			continue
		folder_text = os.listdir(os.path.join(trend_path, author))[1]
		file_text = os.listdir(os.path.join(trend_path, author, folder_text))
		for i in file_text:
			## Remove the .subject 
			temp = "<" + i[:-8] + ">"
			index_list.append(file_name_list.index(temp))

	return index_list

#index_list = find_trend_index_inside_overall_data(trend_path, file_name)
#np.save("trend_index", index_list)
#print(len(index_list))

def load_index(path):
	return np.load(path)


def splitting_dataset(email_data, author_label, index):
	# Training data is larger than the trend techonology provided for me.
	"""
	temp = [i for i in range(len(index))]
	X_train, X_test, train_index, test_temp = train_test_split(index, temp, test_size=0.3, random_state=9)
	X_val, X_test, val_index, test_index = train_test_split(X_test, test_temp, test_size=0.5, random_state=9)
	return X_train, X_val, X_test, train_index, val_index, test_index
	"""
	train_index, temp_index = train_test_split(index, test_size=0.3, random_state=9)
	val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=9)
	train_index = [i for i in range(len(email_data)) if i not in temp_index]
	count = 0
	#print(np.array(email_data)[train_index][3])
	#for i in email_data:
	#	if len(i) > 100 :
	#		count +=1
	return np.array(email_data)[[train_index]], np.array(email_data)[[val_index]], np.array(email_data)[[test_index]], np.array(author_label)[[train_index]], np.array(author_label)[[val_index]], np.array(author_label)[[val_index]]
	#return np.array(email_data)[[train_index]]
trend_index = load_index("trend_index.npy")
#train_data, val_data, test_data = splitting_dataset(email_df, trend_index)


def build_vocab(vocab_path, enron_df):

	if os.path.exists(vocab_path):
		vocab_file = open(vocab_path, 'rb')
		vocab = pickle.load(vocab_file)
		print("Load successfully !!")

	else: 
		word_freq_dict = defaultdict(int)
		all_content = enron_df["Content"].values
		all_subject = enron_df["Subject"].values
		for i in all_content:
			i = i.lower()
			words = word_tokenizer.tokenize(i)
			for word in words:
				word_freq_dict[word] += 1

		for i in all_subject:
			i = i.lower()
			words = word_tokenizer.tokenize(i)
			for word in words:
				word_freq_dict[word] += 1

		vocab = {}
		i = 1
		vocab['UNK'] = 0
		for word, freq in word_freq_dict.items():
			if freq > 5:
				vocab[word] = i
				i += 1

		with open(vocab_path, 'wb') as g:
			pickle.dump(vocab, g)
			print(len(vocab))
			print("vocab save finished")
	return vocab


#vocab = build_vocab("vocab", email_df)


def process_whole_data(data_pickle_path, email_df, author_dict, max_sent_len=100, max_sent_words=30):
	## Remove stopwords using the nltk packages.
	## Truncate the max_sen_len = 100, max_sen_words = 30 
	## Further Improvment: incorporate subject information.
	if not os.path.exists(data_pickle_path):
		eng_stopwords = set(stopwords.words('english'))
		eng_stopwords.add("RE")
		eng_stopwords.add("FW")
		datas = email_df.values
		data_whiten = np.zeros((data.shape[0]), max_sent_len, max_sent_words, dtype=int64)
		label = [] 
		vocab = build_vocab("vocab", email_df)
		UNK = 0
		for line, data in enumerate(datas): 
			content = data[-2]
			label.append(author_dict[data[-1]])
			sents = sent_tokenizer.tokenize(content)
			doc = np.zeros([max_sent_len, max_sent_words])
			for i, sent in enumerate(sents):
				if i < max_sen_len: 
					word_to_index = np.zeros([max_sent_words], dtype=int64)
					temp_word_list = word_tokenizer.tokenize(sent)
					place = 0
					for word in temp_word_list: 
						if place < max_sent_words :
							if word not in eng_stopwords: 
								word_to_index[place] =  vocab.get(word, UNK)
								place += 1 

					"""
					for j, word in enumerate(word_tokenizer.tokenize(sent)):
						if j < max_sent_words:
							if word not in eng_stopwords
							word_to_index[j] = vocab.get(word, UNK)
					"""
					doc[i] = word_to_index
			data_whiten[line] = doc 
		pickle.dump((data_whiten, label), open(data_pickle_path, "wb"))
	else: 
		data_file = open(data_pickle_path, "rb")
		data_whiten, label = pickle.load(data_file)
	return data_whiten, label


def process_whole_data_without_padding(data_pickle_path, email_df, author_dict):
	data_list= []
	if not os.path.exists(data_pickle_path):
		eng_stopwords = set(stopwords.words('english'))
		eng_stopwords.add("RE")
		eng_stopwords.add("FW")
		datas = email_df.values
		label = [] 
		vocab = build_vocab("vocab", email_df)
		UNK = 0
		for line, data in enumerate(datas): 
			content = data[-2].lower()
			author_name = data[-1]
			if author_name in author_dict.keys():
				label.append(author_dict[data[-1]])
			else: 
				if author_name == "harris-s":
					label.append(148)
				elif author_name == "stokley-c":
					label.append(149)
				else: 
					print("Error")
			sents = sent_tokenizer.tokenize(content)
			doc =  [] 
			for i, sent in enumerate(sents):
				temp = []
				temp_word_list = word_tokenizer.tokenize(sent)
				for word in temp_word_list: 
						if word not in eng_stopwords: 
							temp.append(vocab.get(word, UNK))

				doc.append(temp)
			data_list.append(doc)
		pickle.dump((data_list, label), open(data_pickle_path, "wb"))
		print(max(label))
		print(len(data_list))
	else: 

		data_file = open(data_pickle_path, "rb")
		data, label = pickle.load(data_file)
		#print(max(label))
		#print(len(data))
		#print(data[3])
	return data, label

#x_data, y_data = self.process_whole_data("data_path", email_df, author_dict)
#data, label = process_whole_data_without_padding("data", email_df, author_dict)
#train_data, val_data, test_data, train_label, val_label, test_label = splitting_dataset(data, label, trend_index)
#dada = splitting_dataset(data, label, trend_index)

def embedding_matrix(pretrained_vec_path, vaoca_path, email_df, emb_dim):
	embedding_index = {}
	with open(pretrained_vec_path) as f : 
		for line in f : 
			values = line.split()
			word = values[0]
			vec = np.array(values[1:], dtype='float32')
			embedding_index[word] = vec 
	vocab = build_vocab(vaoca_path, email_df)
	emb_matrix = np.random.uniform(-0.5, 0.5, (len(vocab)+1, emb_dim)) / emb_dim
	#emb_matrix = np.random.random(len(vocab)+1, emb_dim)
	for word, i in vocab.items():
		embedding_vec = embedding_index.get(word)
		if embedding_vec is not None: 
			emb_matrix[i] = embedding_vec
	return emb_matrix, (len(vocab)+1)







# Firsy segment the sentence, then delete the stop words.

