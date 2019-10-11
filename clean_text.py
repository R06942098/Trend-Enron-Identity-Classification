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


def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from'),
        'subject': list(email_old["Subject"])
    }

def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results

#email_df = pd.read_csv("enron_all.csv",na_filter= False)
#email = pd.read_csv("emails.csv")
#email_df = pd.DataFrame(parse_into_emails(email.message))
#print(email_df['subject'].values[12])
#print(email_df['body'].values[517385])
#email_df.to_csv("enron_min.csv", index=False)

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
			#k = k.replace("\n", "")
			#k = k.replace("\t", "")
			list_content.append(k)

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





