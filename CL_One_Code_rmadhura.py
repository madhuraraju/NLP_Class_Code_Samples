#################################################
#Author: Madhura Raju
#Objective: Functions provided in inline comments
#################################################

#importing required packages
from __future__ import division
import nltk
import math

#import matplotlib
import matplotlib.pyplot as plt

#importing the corpus
from nltk.corpus import brown

#copying all the words in the Brown corpus
corpus_full_text = brown.words()

#number of words in the corpus
count_words_brown = len(corpus_full_text)

#copying all the words from a particular category
def corpus_text_in_category(category):
    corpus_text_category = list(brown.words(categories=category))
    return corpus_text_category

# returns the number of a particular word in a particular category
def get_words_in_category(word, category):
    no_words_in_cat = corpus_text_in_category(category).count(word)
    return no_words_in_cat

# returns the number of words in a particular category
def get_vocab_size_cat(category):
    cat_vocab = len(brown.words(categories = category))
    return cat_vocab

# 1 counting words:
# 1.1 Probability Distribution of Words
    
def get_prob_word_in_category(word, cat): 
    cat = ""
    if cat=="":
      prob = (corpus_full_text.count(word)/count_words_brown)
    else:
      prob = (get_words_in_category(word,cat)/get_vocab_size_cat(cat))
    ##print 'probability of the word: ' + word 
    ##print '==========================='
    ##print prob
    ## - Added for evaluation:
    ## Need to return values, not print them (-1)
    return prob
  
#1.2 Types and Tokens
#1.2 (a)

def get_vocabulary_size(cat):
    cat = ''
    if cat=='':
       vocab_size = len(corpus_full_text) 
    else:
       vocab_size = get_vocab_size_cat(cat)
    return vocab_size
   
#1.2 (b) function to get the ratio of the tokens to the types

def get_type_token_ratio(cat):
    cat = ''
    if cat=='':
       ratio_type_token = (len(set(corpus_full_text))/len(corpus_full_text))
    else:
       ratio_type_token = (len(set(corpus_text_in_category(cat)))/len(corpus_text_in_category(cat)))
    return ratio_type_token

#1.3 (a) finding the least and the most frequent word in the corpus
def get_top_n_words(numb_words,cat):
    cat = ''
    if cat=='':
       freqd = FreqDist(corpus_full_text)
       distinct_vocab = freqd.keys()
       return distinct_vocab[:numb_words]
    else:
       freqd = FreqDist(corpus_text_in_category(cat))
       distinct_vocab = freqd.keys()
       return distinct_vocab[:numb_words]

def get_bottom_n_words(numb_words,cat):
    cat = "
    if cat=='':
       freqd = FreqDist(corpus_full_text)
       distinct_vocab = freqd.keys()
       return distinct_vocab[-numb_words:]
    else:
       freqd = FreqDist(corpus_text_in_category(cat))
       distinct_vocab = freqd.keys()
       return distinct_vocab[-numb_words:]
 
#1.2 (c) finding the entropy of the given word

def get_entropy(word,cat):
     p_word_here = get_prob_word_in_category(word, cat)
     p_log = float(math.log(p_word_here))
     entropy_word = -(float(p_word_here)*float(p_log))
     print entropy_word 

#2 Context and Similarity
#2.1 (a) Word Contexts
# Get the word contexts

def get_word_contexts(word):
    words_in_cat = corpus_text_in_category('news') 
    if (word == words_in_cat[:1] or word == words_in_cat[-1:]):
       print 'Word has no context in the text'
    else: 
       n = 0
    context_pairs = []
    for w in words_in_cat:
        n = n + 1
        if w == word:

          context_pairs.append(words_in_cat[n-2])
          context_pairs.append(words_in_cat[n])
    i = iter(context_pairs)  
    wp = set(zip(i,i))
    return (wp)
#1.3 (c) Histogram Function

def plot_word_counts():
  #copying all the words in the Brown corpus
  corpus_full_text = brown.words()
  corpus_news = brown.words(categories = 'news')

  fdist = FreqDist(corpus_news)
  xx=fdist.values()
  plt.hist(xx, bins=3000)

  # Annotate the graph 
  plt.xlablel('Frequency of occurences')
  plt.ylabel('Freqency of words in that bucket')
  plt.axis([0,500,0,500]) 
  plt.show()   

#2.1 (b) Unique contexts shared by two words

def get_common_contexts(word1,word2):
    context_words1 = get_word_contexts(word1)
    context_words2 = get_word_contexts(word2)
    
    common_contexts = []    

    for w in context_words1:
        if w in context_words2:
              common_contexts.append(w)
    return common_contexts

#2.1 (c) Average number of shared contexts for each pair of the city names: washington, philadelphia, boston, london
def get_average_shared_contexts(city1,city2):
    con = get_common_contexts(city1,city2)
    count_con = len(con)
    total_con = len(get_word_contexts(city1)) + len(get_word_contexts(city2))
    average = count_con/(total_con-count_con)
    return average*100

# 2.2  Measure Similarity

# 2.2 (a) Creating the feature space given the sentence

def create_feature_space(sentence_list):
    feats = {}
    ctr = 0
    for sent in sentence_list:
         for word in sent.strip().split():
               if not feats.has_key(word):
                   feats[word] = ctr
                   ctr += 1
    return feats  

# 2.2 (a) Vectorizing the sentence given the feature space
def vectorize(feature_space,sentence):
	output_vector = []
	
	for i in range(0,len(feature_space)):
            output_vector.append(0)
       
        for w in sentence.strip().split(' '):
            if feature_space.has_key(w):
                print feature_space[w]
                output_vector[feature_space[w]] = 1	
        return output_vector

#Various Similarity Metrices
# 2.2 (b) Dice Similarity

def dice_similarity(x, y):
    x_len = len(x)
    y_len = len(y)
    
    similar = 0
    num_ones = 0
    

    if x_len != y_len:
        print "length not the same"
        return None
    for ii in xrange(0,x_len):
        if x[ii] == 1 or y[ii] == 1:
            num_ones += 1
            if x[ii] == y[ii]:
                similar += 1
    return (similar * 2) / (x_len + y_len)

# 2.2 (b) Jaccard Similarity

def jaccard_similarity(x, y):
    x_len = len(x)
    y_len = len(y)

    if x_len != y_len:

        print "length not the same"
        return None

    num_ones = 0
    similar = 0

    for ii in xrange(0,x_len):

        if x[ii] == 1 or y[ii] == 1:
            num_ones += 1

            if x[ii] == y[ii]:
                similar += 1

    return float(similar / num_ones)


# 2.2 (b) Cosine Similarity

def dot(x, y):
    return reduce(lambda sum, val: sum + (int(val[0])*int(val[1])),zip(x,y), 0)

def norm(x):
    return math.sqrt(dot(x, x))

def cosine_similarity(x,y):
    x_len = len(x)
    y_len = len(y)
    if x_len != y_len:
        print "length not the same"
        return None 
    return (dot(x,y) / (norm(x) * norm(y)))

#2.2 (e) My metric similarity

def mySimilarity(x,y):
    x1,y1,both=0,0,0
    
    for i in range(len(x)):
        if x[i]!=0: x1+=1 # present in x
        if y[i]!=0: y1+=1 # present in y 
        if x[i]!=0 and y[i]!=0: both+=1 # in both 
    
    return (float(math.pow(both,2))/math.pow((x1+y1-both),2))

# function used to Comparing the metrices with one another:
def compareSimilarityMeasures(sentences):
    
    vector_space = create_feature_space(sentences)
    sentence_feats = []

    for sent in sentences:
        sentence_feats.append(vectorize(vector_space,sent))

    cosine_rank = dict()
    jaccard_rank = dict()
    dice_rank = dict()
    my_rank = dict()
    
    for ii in xrange(0,len(sentence_feats)):

        for jj in xrange(0,len(sentence_feats)):
            
            if jj == ii:
                continue

            print sentences[ii]
            print sentences[jj]

            f1 = sentence_feats[ii]
            f2 = sentence_feats[jj]

            cosine_rank[tuple([ii,jj])] = cosine_similarity(f1,f2)
            jaccard_rank[tuple([ii,jj])] = jaccard_similarity(f1,f2)
            dice_rank[tuple([ii,jj])] = dice_similarity(f1,f2)
            my_rank[tuple([ii,jj])] = mySimilarity(f1,f2)
                
    cosine_ranks = sorted(cosine_rank.items(), key = lambda x: x[1])
    dice_ranks = sorted(dice_rank.items(), key = lambda x: x[1])
    jaccard_ranks = sorted(jaccard_rank.items(), key = lambda x: x[1])
    my_ranks = sorted(my_rank.items(), key = lambda x: x[1])

#print just to make analysis
    print "cosine"
    for k in cosine_ranks:
        print k
    raw_input()
        
    print "dice"
    for k in dice_ranks:
        print k
    raw_input()

    print "jaccard"
    for k in jaccard_ranks:
        print k
    raw_input()

    print "my"
    for k in my_ranks:
        print k
    raw_input()

#3. Modelling word Distributions

#3.1 A sliding window
def make_ngram_tuples(samples,n):

    sample_len = len(samples)
    ngram_list = []

    for i in range(max(0,sample_len - n + 1)):
        
        if n==1:
            ngram_list.append(tuple([None,samples[i]]))
        else:
            ngram_list.append(tuple([tuple(samples[i:i+n-1]),samples[i+n-1]]))
    return ngram_list

#3.3 Generating Text 
class NGramModel:

       Ngrams = []
       NgramProb = {}
       NgramCondFreq = {}

       def __init__(self,training_data, n):
               self.Ngrams = make_ngram_tuples(training_data,n)
               self.NgramCondFreq = nltk.ConditionalFreqDist(self.Ngrams)

       def prob(self,context,event):
               return (self.NgramCondFreq[context][event] / self.NgramCondFreq[context].N())


#test and execution of the code- Could contain calling the same function with different parameter values merely for checking edge cases
if __name__ == '__main__':
   # get_prob_word_in_category('president','government')
   # get_vocabulary_size('news')
   # get_type_token_ratio('')
   # get_top_n_words(30,'')
   # get_bottom_n_words(30,'')
   # get_entropy('the','editorial')
   # get_word_contexts('the')    
   # get_common_contexts('he','said')
   # get_average_shared_contexts('Philadelphia','Washington')
   # get_average_shared_contexts('Philadelphia','Boston')
   # get_average_shared_contexts('Philadelphia','London')
   # get_average_shared_contexts('Boston','Washington')
   # get_average_shared_contexts('London','Boston')
   # get_average_shared_contexts('Washington','London')
   #  get_average_shared_contexts('Philadelphia','Washington')
   # sentences = ["the quick brown fox jumped over the lazy dog","the lazy dog the lazy dog"]; 
   # sent_sim = ["the moon is the moon","the moon is the sun","the sun is the sun","sun could be moon","moon can be sun","sun and the moon"]
   # compareSimilarityMeasures(sent_sim)
   # feature_space = create_feature_space(sentences)
   # v1 =  vectorize(feature_space,"the fox jumped on the lazy dog") 
   # v2 =  vectorize(feature_space,"deepak fox")
   # print "cosine :",cosine_similarity(v1,v2)
   # print "jaccard :",jaccard_similarity(v1,v2)
   # print "dice :",dice_similarity(v1,v2)
   # print "My:", mySimilarity(v1,v2)
   # words = ['brown','fox','jumped','over','the','lazy','brown','deepak'];
   # lang_model = NGramModel(words,2)
   # print lang_model.prob(('brown',),'fox')
   # plot_word_counts()
