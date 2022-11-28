import os
import collections
import pickle
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re
from bert import tokenization

checkpoint_path = './chinese_L-12_H-768_A-12/model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=False)
#insert words of a file
def insert_word(f):
	global all_words
	for l in f:
		words=re.split('\s|-',l.lower().split("|||")[0].strip())
		
		all_words+=words

#convert words to numbers
def convert_words_to_number(f, dataset, labels):
	global common_word
	for l in f:
		# try:
			# words=re.split('\s|-',l.lower().split("|||")[0].strip())
			words = l.split("|||")[0].strip('\n')
			label=l.lower().split("|||")[1].strip('\n')
			# words=[common_word[w] if w in common_word else 1 for w in words]

			tokens = tokenizer.tokenize(words)
			tokens = ['[CLS]'] + tokens + ['[SEP]']
			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			while len(input_ids) < 30:
				input_ids.append(0)
			if len(input_ids) > 30:
				input_ids = input_ids[:30]

			dataset+=[input_ids]
			labels+=[label]
		# except:
		# 	continue

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
print(common_word)
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
		pickle.dump(((train,train_label) ,(test,test_label), (valid,valid_label)), open(location+file+'/dataset.mybert', 'wb'))


#create embedding vector matrix
# word_vectors = KeyedVectors.load_word2vec_format('vectors.gz', binary=True)
word2vec=[[0]*768, [0]*768]
for number, word in sorted(zip(common_word.values(), common_word.keys())):
	# try:
	# 	print(type(word_vectors.word_vec(word)))
	# 	word2vec.append(word_vectors.word_vec(word).tolist())
	# except KeyError:
	# 	print(word+ " not found")
	word2vec.append([0]*768)
pickle.dump(word2vec, open('vectors', 'wb'))
print(len(word2vec))
