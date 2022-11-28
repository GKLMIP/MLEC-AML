import os
import collections
import pickle
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re
import nltk

#insert words of a file
def insert_word(f):
	global all_words
	for l in f:
		words=re.split('\s|-',l.lower().split("|||")[0].strip())
		# words = nltk.word_tokenize(l.lower().split("|||")[0].strip())
		# words = [i for i in words]
		all_words+=words

#convert words to numbers
def convert_words_to_number(f, dataset, labels):
	global common_word
	for l in f:
		try:
			words=re.split('\s|-',l.lower().split("|||")[0].strip())
			print(words)
			# words = nltk.word_tokenize(l.lower().split("|||")[0].strip())
			# words = [i for i in words]
			label=l.lower().split("|||")[1].strip()
			words=[common_word[w] if w in common_word else 1 for w in words]
			dataset+=[words]
			if label == '':
				label = l.lower().split("|||")[2].strip()
			labels += [label]
			# print(type(label))
		except Exception as e:
			print(e)
			continue
	print(set(labels))
vocab=50000
gap=2
vocab_size=vocab-2
location='./chi_augmentation/'
all_words=[]

#iterate all files
for file in os.listdir(location):
	if file != '.DS_Store':
		with open(location+file+"/trn") as f:
			insert_word(f)
		with open(location+file+"/dev") as f:
			insert_word(f)
		with open(location+file+"/tst") as f:
			insert_word(f)

#take out frequent words 
counter=collections.Counter(all_words)
common_word=dict(counter.most_common(vocab_size))

#number them
c=2
for key in common_word:
	common_word[key]=c
	c+=1
print(len(common_word))
pickle.dump(common_word, open('dictionary', 'wb'))

for file in os.listdir(location):

	if file != '.DS_Store':
		train=[]
		train_label=[]
		test=[]
		test_label=[]
		valid = []
		valid_label = []
		with open(location+file+"/trn") as f:
			convert_words_to_number(f, train, train_label)
		with open(location+file+"/dev") as f:
			convert_words_to_number(f, valid, valid_label)
		with open(location+file+"/tst") as f:
			convert_words_to_number(f, test, test_label)
		pickle.dump(((train,train_label) ,(test,test_label), (valid,valid_label)), open(location+file+'/dataset', 'wb'))


#create embedding vector matrix
# word_vectors = KeyedVectors.load_word2vec_format('vectors.gz', binary=True)
word2vec=[[0]*128, [0]*128]
for number, word in sorted(zip(common_word.values(), common_word.keys())):
	# try:
	# 	print(type(word_vectors.word_vec(word)))
	# 	word2vec.append(word_vectors.word_vec(word).tolist())
	# except KeyError:
	# 	print(word+ " not found")
	word2vec.append([0]*128)
pickle.dump(word2vec, open('vectors', 'wb'))
print(len(word2vec))
