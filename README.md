# Enron Email Identity Classification

## **Data**
Because some of the data (content) from trend tech  are empty files, I download the complete Enron email dataset from the [Kaggle link](https://www.kaggle.com/wcukierski/enron-email-dataset). 
### Preprocess
```
python data_prepro.py
```
I execute this script to get the index of 112000 samples from the dataset provided from trend tech.  Meanwhile, these 112000 pieces of data is splitted into training set, validation set and testing set. It is worth mentioning that the training set is combined with the data I download from kaggle, note that it is not overlapped with validation/testing set.

### Tokenize
```
Usw NLTK package.
```
### Pretrained Embedding 
[GloVe Model pretrained on Wikipedia dataset](https://github.com/stanfordnlp/GloVe)


## **Model**
I regard this problem as the document classification problem (title v.s subject while documnet v.s. email content) so that I choose the [HAN](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) to implement. 

Some of the ideas and future works are listed in the figure below:
![image](https://github.com/R06942098/Trend-Takehome-Assignment/blob/master/img/trend.png)

## **Train the Model**
```
python main_HAN.py --train True 
```
Some paratmeters whcih can be tuned are listed in this filem, too.
## **Evaluation and Future Works**
I don't have the comprehensive results now, because machine learns nothing with the data preprocessed by me. I find several problems and it can be done duing my next military vocation (9/27~9/29): 

- The NLTK word tokenize seems to have some problems: 
   - Large dictionary: the words segmented from this packages are sometimes not formal words...
- Concatenate TF-IDF values the the embedding vector and Stylometric vecot (put emphasize some high frequence words used for certain authors)
- Concatenate stylometric vectors to the latent vector . Moreover, we can search more stylometric... 
- Survey state-of-the-art paper of the document classification prbolem and some lauguage data analysis technique.
- If failing again, I may try some retrieval-based models. 

## **Reference**
[Github](https://github.com/tqtg/hierarchical-attention-networks)
