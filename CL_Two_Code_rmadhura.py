##################################################################################################
#Author: Madhura Raju
#Objective: The program performs Topic Classification based on various features that are generated from the corpus
#         : Functions provided in inline comments
##################################################################################################

#importing required packages
from __future__ import division
import nltk
import math
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist, ConditionalFreqDist

#Computes Features based on overall properties of the data
def get_coarse_level_features(dataset, output_file):
# accessing the corpus
    corpus_root = '/home1/c/cis530/data-hw2/' 
    dataset_path = corpus_root + dataset

# Reading the files from the directories
    files = PlaintextCorpusReader(dataset_path, '.*')
    ids = files.fileids()
    stopFile = PlaintextCorpusReader(corpus_root, 'stopwlist.txt')
    stops = stopFile.words()

#Opening a file that has to be written to
    out = open(output_file, 'w')

    for i in range(0,len(ids) - 1):
#Initializing certain variables
        tokens_count=0
        types = 0
        non_stops_count=0
        sents_count = 0
        avg_sent_len=0
        cap_count = 0

        tokens=files.words(ids[i])
#Computing Number of Tokens
        tokens_count = len(tokens)

#Computing Number of types
        types = len(set(tokens))
        non_stops=[]

#Computing Number of Content Words
        for t in tokens:
            if t not in stops:
                non_stops.append(t)
        non_stops_count = len(non_stops)

#Finding Average Sentence Length
        sent = []
        sent = files.sents(ids[i])
        sents_count = len(sent)
        sent_len=0
        for s in sent:
            sent_len = sent_len + len(s)
        avg_sent_len = sent_len/float(sents_count)

#Computing Number of Captilized Words
        for c in non_stops:
            if c.istitle():
                cap_count = cap_count+1
        current_file = dataset + '/' + ids[i]
        e = current_file.split('/')
        out.write(current_file +' '+ e[-2] + ' tok:' + str(tokens_count) + ' typ:' + \
str(types) + ' con:' + str(non_stops_count) + ' sen:' + str(sents_count) + ' len:' + str(avg_sent_len) + ' cap:' + str(cap_count)+ '\n')
        out.flush()

#Test-Cases:
#get_coarse_level_features('Test_set', 'Test_set.coarseFeatures')
#get_coarse_level_features('Training_set_small', 'Training_set_small.coarseFeatures')
#get_coarse_level_features('Training_set_large', 'Training_set_large.coarseFeatures')  
 
#Adding to a file each of the features that were created in a tabular format
     
def prepare_pos_features(Language_model_set, output_file):
    corpus_root = '/home1/c/cis530/data-hw2/' + Language_model_set
    texts = PlaintextCorpusReader(corpus_root, '.*')
    text = texts.words()
    tagged_text = nltk.pos_tag(text)
    merged_tag_text = mergeTags(tagged_text)
    lists = seperate_pos(merged_tag_text)
    nouns_dist = FreqDist(lists[0])
    top_nouns = nouns_dist.keys()[:200]
    verbs_dist = FreqDist(lists[1])
    top_verbs =verbs_dist.keys()[:200]
    advs_dist = FreqDist(lists[2])
    top_advs =advs_dist.keys()[:100]
    prep_dist = FreqDist(lists[3])
    top_preps =prep_dist.keys()[:100]
    adjs_dist = FreqDist(lists[4])
    top_adjs =adjs_dist.keys()[:200]


    out = open(output_file, 'w')

    for n in top_nouns:
        out.write('NN'+ n + '\n')
    for v in top_verbs:
        out.write('VV'+ v + '\n')
    for av in top_advs:
        out.write('ADV'+ av + '\n')
    for p in top_preps:
        out.write('PREP'+ p + '\n')
    for aj in top_adjs:
        out.write('ADJ'+ aj + '\n')

#Generating POS
def seperate_pos(text):
    nouns = []
    verbs = []
    adjs = []
    advs = []
    preps = []
    lists = []
    for el in text:
        if (el[1] == 'NN'):
            nouns.append(el[0])
        elif (el[1] == 'VV'):
            verbs.append(el[0])
        elif (el[1] == 'ADJ'):
            adjs.append(el[0])
        elif (el[1] == 'ADV'):
            advs.append(el[0])
        elif(el[1] == 'PREP'):
            preps.append(el[0])
    lists.append(nouns)
    lists.append(verbs)
    lists.append(advs)
    lists.append(preps)
    lists.append(adjs)
    return lists

#merging the sentences according to their POS
def mergeTags(text):
    Nouns = ['NN','NNS', 'NNP', 'NNPS']
    Verbs = ['VB', 'VBD', 'VBG', 'VBN','VBP','VBZ']
    Adjs = ['JJ', 'JJR', 'JJS']
    Advs = ['RB','RBR','RBS']
    Prep = ['IN']

    N = 'NN'
    V = 'VV'
    Aj = 'ADJ'
    Av = 'ADV'
    P = 'PREP'

    merged = []

    for el in text:
        if(el[1] in Nouns):
            merged.append((el[0],N))
        elif(el[1] in Verbs):
            merged.append((el[0],V))
        elif(el[1] in Adjs):
            merged.append((el[0],Aj))
        elif(el[1] in Advs):
            merged.append((el[0],Av))
        elif(el[1] in Prep):
            merged.append((el[0],P))
        else: merged.append(el)
    return merged
                     

def get_pos_features(dataset, feature_set_file, output_file):
    corpus_root = '/home1/c/cis530/data-hw2/' 
    dataset_path = corpus_root + dataset
    files = PlaintextCorpusReader(dataset_path, '.*')
    ids = files.fileids()
    feature_set_tuples = get_feature_set_tuples(feature_set_file)
    out_file = open(output_file,'w')
    ## Off by one error here, don't use range
    for i in range(len(ids)):
        out_string= ''
        current_file = dataset + '/'+ids[i]
        e = current_file.split('/')
        out_string = out_string + current_file+ ' '+e[-2]
        tagged_file = nltk.pos_tag(files.words(ids[i]))
        for feature in feature_set_tuples:
            count =0
            for tag in tagged_file:
                if(feature == tag):
                    count = count +1
            out_string = out_string + " " + feature[1]+feature[0]+ ':' + str(count)
        out_file.write(out_string + '\n')
        out_file.flush()

#Combining the Features in a particular format after generating the feature words based on POS: 
def get_feature_set_tuples(feature_file):
    tuple_list = []
    path = feature_file.split('/')
    file_name = path[(-1):]
    root = ''
    for i in range(len(path)-1):
        root=root + path[i] + '/'
    feature_file = PlaintextCorpusReader(root, file_name)
    features = feature_file.words()
    for each in features:
        if(each[:2] == 'NN' or each[:2] =='VV'):
            tuple_list.append((each[2:], each[:2]))
        elif(each[:3] =='ADJ' or each[:3]=='ADV'):     
            tuple_list.append((each[3:], each[:3]))
        elif(each[:4]=='PREP'):
            tuple_list.append((each[4:], each[:4]))
    return tuple_list

   
def _sliding_window(l, n):
    return [tuple(l[i:i+n]) for i in range(len(l)-n+1)]

def make_ngram_tuples(l, n):
     t = _sliding_window(l, n)
     if n == 1:
          return [((None,), s) for (s,) in t]
     return [(tuple(s[:-1]), s[-1]) for s in t]

#Generating NGramModels

class NGramModel:
     def __init__(self, training_data, n):
         self.trainData = training_data
         self.trainData.append('UNK')
         ngrams = make_ngram_tuples(self.trainData, n)
         self.cfd = ConditionalFreqDist(ngrams)
         self.n = n

     def _get_next_word(self, context):
          freq_dist = self.cfd[context]
          r = random.random()
          m = 0.0
          for s in freq_dist.samples():
               if (m <= r) and (r < (m + freq_dist.freq(s))):
                    return s
               m += freq_dist.freq(s)
          return None

     def generate(self, n, context):
          words = []
          if self.n > 1:
              words = list(context)
          for i in range(n - len(words)):
               next_word = self._get_next_word(context)
               words.append(next_word)
               if self.n > 1:
                   context = tuple(context[1:] + (next_word,))
          return words
     def prob(self, context, event):
         return (self.cfd[context][event]+1)/float(sum((self.cfd[context].values()))+ len(set(self.trainData)))

#generating the LM features

def get_lm_features(dataset, output_file):      
    corpus_root = '/home1/c/cis530/data-hw2/'
    bigram_root = corpus_root + 'Language_model_set/'

    fin_files = PlaintextCorpusReader(bigram_root+'Finance/','.*')
    fin_words = list(fin_files.words())
    fin_model = NGramModel(fin_words, 2)

    health_files = PlaintextCorpusReader(bigram_root+'Health/','.*')
    health_words = list(health_files.words())
    health_model = NGramModel(health_words, 2)

    res_files = PlaintextCorpusReader(bigram_root+'Research/','.*')
    res_words = list(res_files.words())
    res_model = NGramModel(res_words, 2)

    com_files = PlaintextCorpusReader(bigram_root+'Computers_and_the_Internet/','.*')
    com_words = list(com_files.words())
    com_model = NGramModel(com_words, 2)

    test_files = PlaintextCorpusReader(corpus_root+dataset, '.*')
    ids = test_files.fileids()

    out_file = open(output_file,'w')

    for j in range(0,len(ids)):
        file_words = test_files.words(ids[j])
        out_str = ''
        current_file = dataset + '/'+ids[j]
        e = current_file.split('/')
        out_str = out_str + current_file+ ' '+e[-2]
        sum_fin=0
        sum_health=0
        sum_res=0
        sum_com=0                                                                         
        text_len = len(file_words)
        for i in range(1,len(file_words)):
            sum_fin = sum_fin + math.log(fin_model.prob((file_words[i-1],),file_words[i]))
            comp_fin = float((-sum_fin)*(1/float(text_len)))
            sum_health = sum_health + math.log(health_model.prob((file_words[i-1],),file_words[i]))

            comp_health = (float(-sum_health))*(1/float(text_len))
            sum_res = sum_res + math.log(res_model.prob((file_words[i-1],),file_words[i]))
            comp_res = (float(-sum_res))*(1/float(text_len))
            sum_com = sum_com + math.log(com_model.prob((file_words[i-1],),file_words[i])) 
            comp_com = (float(-sum_com))*(1/float(text_len))
            out_str = out_str + ' finprob:'+str(round(sum_fin,2))+' hlprob:'+str(round(sum_health,2))+' resprob:'\
+str(round(sum_res,2))+ ' coprob:' + str(round(sum_com,2)) + ' finper:' + str(round(comp_fin,2)) + ' hlper:'+\
str(round(comp_health,2))+ ' resper:' + str(round(comp_res,2)) + ' coper:' + str(round(comp_com,2)) 
           out_file.write(out_str + '\n')
           out_file.flush()

#generating the Ngram Models and the Language Models for the dataset

class NGramModel2:

       Ngrams = []
       NFreqDist = {}
       NgramProb = {}
       NgramCondFreq = {}

       def __init__(self,training_data, n):
               self.Ngrams = make_ngram_tuples(training_data,n)
               self.NgramCondFreq = nltk.ConditionalFreqDist(self.Ngrams)
	       self.NFreqDist = nltk.FreqDist(training_data)
	       self.Tot = len(set(training_data))
       def prob(self,context,event):
               return (self.NgramCondFreq[context][event]+1 / (self.NgramCondFreq[context].N() + self.Tot))

def create_LM_on_dataset(dataset):
    
     corpus_root = '/home1/c/cis530/data-hw2/Language_model_set/'
     dataset_path = corpus_root + dataset
     files = PlaintextCorpusReader(dataset_path, '.*')
     ids = files.fileids()
     for i in range(len(ids)):
         words = files.words(ids[i])
     lang_model = NGramModel2(words,2)
     
     return lang_model

#2.1
#Finding the best Fit for a sentence from a list of words
def get_fit_for_word(sentence, word, langmodel):

    lang_model = create_LM_on_dataset(langmodel)
    log_prob = 0 
    sent = sentence.strip().split(' ')
    mod_sent = []
    for w in sent:
        if w == '<blank>':
		mod_sent.append(word)
	else:
		mod_sent.append(w)
    sent_bigrams = make_ngram_tuples(mod_sent,2)
    
    for bigram in sent_bigrams:
	log_prob += math.log(lang_model.prob(bigram[0],bigram[1]))

    return log_prob

#2.2
def get_bestfit_topic(sentence,wordlist,topic):
    log_wd = {}
    for wd in wordlist:
        log_wd[wd]=get_fit_for_word(sentence,wd,topic)
    min_log=min(log_wd.iteritems(),key=operator.itemgetter(1))
    best_fit_word = min_log[0]
    return best_fit_word

def get_feature_file(features_to_use, output_file):
    out_file = open(output_file, 'w')
    file1=open(features_to_use[0],'r')
    file2 = open(features_to_use[1],'r')
    file3_bool=False
    if(len(features_to_use)>2):
	file3=open(features_to_use[2])
	file3_bool=True
    read_file1 = file1.readlines()
    read_file2 = file2.readlines()
    if file3_bool:
	read_file3=file3.readlines()
    for i in range(len(read_file1)):
	line1 = read_file1[i]
	line2 = read_file2[i]
	line_1_tokens = line1.split()
	rep_text = line_1_tokens[0] + ' ' + line_1_tokens[1]
	line2 = line2.replace(rep_text, '')
	final_line = line1.strip() + line2.strip()
	if file3_bool:
		line3 = read_file3[i]
		line3 = line3.replace(rep_text, '')
		final_line = final_line + line3
	out_file.write(final_line + '\n')

#Training the Naive Bayes from the NLTK library.
	
def get_NB_classifier(training_examples):
    fip = open(training_examples, 'r')
    features = fip.readlines()
    tup_list = []
    actual_features = []
    af_dict = {}
    lines_split = []
    for lines in features:
        lines_split = lines.split()
        rep = lines_split[0] + ' ' + lines_split[1]
	lines = lines.replace(rep,'')
    	topic = lines_split[1]
        for fv_pair in lines.split():
            vals = fv_pair.split(':')
            af_dict[vals[0]] = vals[1]
            tup_list.append((af_dict,topic))
   #input to the classifier: tup_list
    classifier = nltk.NaiveBayesClassifier.train(tup_list)
    return classifier

#Performing Classification of the Documents based on the features that were already extracted

def classify_documents(test_examples, model, classifier_output):
    ftp = open(test_examples,'r')
    features = ftp.readlines()
    tup_test_list = []
    actual_test_features = []
    lines_split = []
    feat_dict = {}
    fout = open(classifier_output,'w')
    for lines in features:
        lines_split = lines.split()
        rep = lines_split[0] + ' ' + lines_split[1]
	lines = lines.replace(rep,'')
        topic = lines_split[1]
        print lines
        for fv_pair in lines.split():
            vals = fv_pair.split(':')
            feat_dict[vals[0]] = vals[1]
            classified_output = model.classify(feat_dict)
        fout.write(lines_split[0] + " " + topic + " " + classified_output + "\n")
        fout.flush()
# required for the Write up            
    accuracy_result = nltk.classify.accuracy(model, tup_test_list)
    return accuracy_result 

##model1 = get_NB_classifier('Training_set_small.coarseFeatures')
##classify_documents('Test_set.coarseFeatures', model1, 'Training_coarse.results')
#get_feature_file(['Test_set.coarseFeatures', 'Test_set.posfeatures_2'], 'test_pos_coarse.txt')	
