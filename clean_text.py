import os
import pickle
import nltk
import re
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from collections import defaultdict
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

clean = True

if clean:
	email_file = pd.read_csv("enron_all.csv", na_filter= False)
	subject = email_file[["Subject"]].values
	user = email_file[["User"]].values
	content = email_file[["Content"]].values
	from_user = email_file[["From"]].values

	list_subject = [] 
	list_user = []
	list_content = []
	list_from = []
	for i, j, k, l in zip(user, subject, content, from_user):
		list_subject.append(j[0])
		list_user.append(i[0])
		list_from.append(l[0])
		### filter... Forwarded 
		k = k[0].split("---------------------- Forwarded")[0]
		k = k.split(" -----Original Message-----\n")[0]
		if k in ["", "\n\n"]:
			list_content.append("Empty")
		else:
			list_content.append(k.split(" -----Original Message-----\n")[0])

	email_dict = {}
	email_dict["subject"] = list_subject
	email_dict["content"] = list_content
	email_dict["user_id"] = list_user
	email_dict["from"] = list_from
	email_df = pd.DataFrame(email_dict)
	email_df.to_csv("enron_clean.csv", index=False)

else: 
	# count the content matching the from_user:
	email_file = pd.read_csv("enron_clean.csv", na_filter= False)
	subject = email_file[["subject"]].values
	content = email_file[["content"]].values
	user_id = email_file[["user_id"]].values
	user_from = email_file[["from"]].values

	count = 0
	index = 0
	for i, j in zip(user_id, user_from):
		user_name = i[0]
		from_name = j[0].split("@")[0]
		if user_name == 'mims-thurston-p':
			user_name_list = user_name.split("-")
			from_name_list = from_name.split(".")
			from_name_list.reverse()
			print(index)
			if from_name_list[0] == user_name_list[0]:
				#if from_name_list[2] == "l":
				#	count += 1
				count += 1

		else:
			user_name_list = user_name.split("-")
			from_name_list = from_name.split(".")
			from_name_list.reverse()
			print(index)
			if from_name_list[0] == user_name_list[0]:
				#if from_name_list[1][0] == user_name_list[1]:
				#	count += 1 
				count += 1 
		index += 1 

	print(count)





