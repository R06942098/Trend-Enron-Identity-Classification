import pandas as pd 
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import re 
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download()
ENRON_EMAIL_DATASET_PATH = "emails.csv"
emails_df = pd.read_csv(ENRON_EMAIL_DATASET_PATH)
print(emails_df.shape)
emails_df.head()

#########################################################
# Sort out required email features: date, subject, content
#########################################################

# source https://www.kaggle.com/zichen/explore-enron
## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

import email
# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]


eng_stopwords = set(stopwords.words('english'))

for i in  ['!',',','.','?','-s','-ly','</s>','s']: 
	eng_stopwords.add(i)


def clean_text(text):
    #text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)


drop_index_list = ["Mime-Version", "Content-Type",'Content-Transfer-Encoding']
# Parse content from emails
emails_df['Content'] = list(map(get_text_from_email, messages))
emails_df["User"] = emails_df['file'].map(lambda x: x.split("/")[0])
#emails_df["clean_content"] = emails_df.Content.apply(clean_text)
### Option
emails_df.drop(drop_index_list, axis=1)
emails_df.to_csv("enron_all.csv", index=False)


