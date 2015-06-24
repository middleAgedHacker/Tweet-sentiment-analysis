# this one is for getting topics from twitter
import re
import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
import lda
import lda.datasets
import cPickle
#from nltk import bigrams

c = 0
numBdrs = np.zeros((28474, 1), dtype = np.int)
startFlag = True
tweetList = []
lineBuf = ''


with open('TweetDataset.txt') as f:
    for count, line in enumerate(f):
        indx = [m.start() for m in re.finditer("'''", line)]
        #numBdrs[count] = len(indx)
        numBdrs = len(indx)
        if numBdrs == 0:
            lineBuf = lineBuf + line
        elif numBdrs == 1:
            if startFlag:
                lineBuf = line[indx[0]+3:]
                startFlag = False
            else:
                lineBuf = lineBuf + line[:indx[0]]
                tweetList.append(lineBuf)
                startFlag = True
        else:
            if startFlag:
                lineBuf = line[indx[0]+3:indx[1]]
                tweetList.append(lineBuf)
            else:
                lineBuf = lineBuf + line[:indx[0]]
                tweetList.append(lineBuf)
                lineBuf = line[indx[1]+3:]
                


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    #r'^https?:\/\/.*[\r\n]*', #urls tho change
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'RT', 'via', 'I', 'amp', r'^https?:\/\/.*[\r\n]*']
count_all = Counter()
count_nonhash = Counter()
#count_bigram = Counter()
vocabL = cPickle.load(open('saveVocab.p', 'rb'))
cskInd = np.zeros((20623), dtype=np.bool)
muInd = np.zeros((20623), dtype=np.bool)
afcInd = np.zeros((20623), dtype=np.bool)
nyInd = np.zeros((20623), dtype=np.bool)
mcInd = np.zeros((20623), dtype=np.bool)
chlInd = np.zeros((20623), dtype=np.bool)
kkrInd = np.zeros((20623), dtype=np.bool)
rmInd = np.zeros((20623), dtype=np.bool)
lfcInd = np.zeros((20623), dtype=np.bool)

cskTweetTokens = []
afcTweetTokens = []
muTweetTokens = []
nyTweetTokens = []
mcTweetTokens = []
chlTweetTokens = []
kkrTweetTokens = []
rmTweetTokens = []
lfcTweetTokens = []

for tweetNum, teet in enumerate(tweetList):
    teetChars = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', teet)
    teetChars2 = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', teetChars) # removing links
    teetChars3 = re.sub(r'<[^>]+>', '', teetChars2) # removing html tags
    teetChars4 = re.sub(r'(?:@[\w_]+)', '', teetChars3) # removing mentions
    teetChars5 = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', teetChars4) # removing numbers
    #teetChars6 = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', teetChars5) # removing links
    tweetTokens = preprocess(teetChars5, lowercase=True)
    termsWOstop = [term for term in tweetTokens if term not in stop]
    terms_nonhash = [term for term in termsWOstop if not term.startswith('#')]
    count_all.update(termsWOstop)
    if '#csk' in termsWOstop:
        cskInd[tweetNum] = True
        cskTweetTokens.append(terms_nonhash)
    if ('#afc' or '#arsenalfc' or '#gunners' or '#coyg') in termsWOstop:
        afcInd[tweetNum] = True
        afcTweetTokens.append(terms_nonhash)
    if ('#mufc' or '#manutd') in termsWOstop:
        muInd[tweetNum] = True
        muTweetTokens.append(terms_nonhash)
    if ('#yankees' or '#newyorkyankees') in termsWOstop:
        nyInd[tweetNum] = True
        nyTweetTokens.append(terms_nonhash)
    if ('#realmadrid' or '#halamadrid') in termsWOstop:
        rmInd[tweetNum] = True
        rmTweetTokens.append(terms_nonhash)
    if ('#mcfc' or '#manchestercity' or '#mancity') in termsWOstop:
        mcInd[tweetNum] = True
        mcTweetTokens.append(terms_nonhash)
    if ('#chelsea' or '#cfc' or '#chelseafc') in termsWOstop:
        chlInd[tweetNum] = True
        chlTweetTokens.append(terms_nonhash)
    if '#kkr' in termsWOstop:
        kkrInd[tweetNum] = True
        kkrTweetTokens.append(terms_nonhash)
    if '#lfc' in termsWOstop:
        lfcInd[tweetNum] = True
        lfcTweetTokens.append(terms_nonhash)
    #count_nonhash.update(terms_nonhash)
    #terms_bigram = bigrams(termsWOstop)
    #[count_bigram.update(list(t)) for t in terms_bigram]
    
"""
cskNi = np.array(np.nonzero(cskInd))
cskIndRel = cskNi.squeeze()
tweetAr = np.array(tweetList)
cskTweets = tweetAr[cskIndRel]

afcNi = np.array(np.nonzero(afcInd))
afcIndRel = afcNi.squeeze()
afcTweets = tweetAr[afcIndRel]

muNi = np.array(np.nonzero(muInd))
muIndRel = muNi.squeeze()
muTweets = tweetAr[muIndRel]
"""


"""
vocabList = list(vocab)
vocab.update(set(terms_nonhash))
cPickle.dump(vocabList, open('saveVocab.p', 'wb'))
"""


"""this is the LDA part"""
"""
L = np.shape(cskTweetTokens)
cskX = np.zeros((L[0], 9703), dtype = np.int16)
for tweetNum, teet in enumerate(cskTweetTokens):
    cskX[tweetNum, [vocabL.index(term) for term in teet]] = 1
    
L = np.shape(muTweetTokens)
muX = np.zeros((L[0], 9703), dtype = np.int16)
for tweetNum, teet in enumerate(muTweetTokens):
    muX[tweetNum, [vocabL.index(term) for term in teet]] = 1

L = np.shape(afcTweetTokens)
afcX = np.zeros((L[0], 9703), dtype = np.int16)
for tweetNum, teet in enumerate(afcTweetTokens):
    afcX[tweetNum, [vocabL.index(term) for term in teet]] = 1
    
vocabTup = tuple(vocabL)
n_top_words = 4
modelcsk = lda.LDA(n_topics=5, n_iter=50, random_state=5)
modelcsk.fit(cskX)
topic_word = modelcsk.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocabTup)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
modelafc = lda.LDA(n_topics=5, n_iter=50, random_state=5)
modelafc.fit(afcX)
topic_word = modelafc.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocabTup)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


modelmu = lda.LDA(n_topics=5, n_iter=50, random_state=5)
modelmu.fit(muX)
topic_word = modelmu.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocabTup)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
"""

# training part !

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs
    
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

from sklearn.ensemble import RandomForestRegressor
forestReg = RandomForestRegressor( n_estimators = 100 )

forestReg = forestReg.fit( trainDataVecs, train["sentiment"] )

TweetDataVecsCsk = getAvgFeatureVecs( cskTweetTokens, model, num_features)
resCsk = forestReg.predict(TweetDataVecsCsk)

TweetDataVecsAfc = getAvgFeatureVecs( afcTweetTokens, model, num_features)
resAfc = forestReg.predict(TweetDataVecsAfc)

TweetDataVecsMu = getAvgFeatureVecs( muTweetTokens, model, num_features)
resMu = forestReg.predict(TweetDataVecsMu)

TweetDataVecsNy = getAvgFeatureVecs( nyTweetTokens, model, num_features)
resNy = forestReg.predict(TweetDataVecsNy)

TweetDataVecsMc = getAvgFeatureVecs( mcTweetTokens, model, num_features)
resMc = forestReg.predict(TweetDataVecsMc)

TweetDataVecsChl = getAvgFeatureVecs( chlTweetTokens, model, num_features)
resChl = forestReg.predict(TweetDataVecsChl)

TweetDataVecsKkr = getAvgFeatureVecs( kkrTweetTokens, model, num_features)
resKkr = forestReg.predict(TweetDataVecsKkr)

TweetDataVecsRm = getAvgFeatureVecs( rmTweetTokens, model, num_features)
resRm = forestReg.predict(TweetDataVecsRm)

TweetDataVecsLfc = getAvgFeatureVecs( lfcTweetTokens, model, num_features)
resLfc = forestReg.predict(TweetDataVecsLfc)


cskIndArr = cskInd.nonzero()
afcIndArr = afcInd.nonzero()
muIndArr = muInd.nonzero()
nyIndArr = nyInd.nonzero()
mcIndArr = mcInd.nonzero()
chlIndArr = chlInd.nonzero()
kkrIndArr = kkrInd.nonzero()
rmIndArr = rmInd.nonzero()
lfcIndArr = lfcInd.nonzero()

print 'negative csk \n\n'
tList = np.nonzero(resCsk < np.percentile(resCsk, 1))
for ind in tList[0]:
    print tweetList[cskIndArr[0][ind]] + '\n'
    
print 'positive csk \n\n'
tList = np.nonzero(resCsk > np.percentile(resCsk, 99))
for ind in tList[0]:
    print tweetList[cskIndArr[0][ind]] + '\n'


print 'negative afc \n\n'
tList = np.nonzero(resAfc < np.percentile(resAfc, 1))
for ind in tList[0]:
    print tweetList[afcIndArr[0][ind]] + '\n'
    
print 'positive afc \n\n'
tList = np.nonzero(resAfc > np.percentile(resAfc, 99))
for ind in tList[0]:
    print tweetList[afcIndArr[0][ind]] + '\n'



print 'negative mu \n\n'
tList = np.nonzero(resMu < np.percentile(resMu, 1))
for ind in tList[0]:
    print tweetList[muIndArr[0][ind]] + '\n'
    
print 'positive mu \n\n'
tList = np.nonzero(resMu > np.percentile(resMu, 99))
for ind in tList[0]:
    print tweetList[muIndArr[0][ind]] + '\n'



print 'negative ny \n\n'
tList = np.nonzero(resNy < np.percentile(resNy, 1))
for ind in tList[0]:
    print tweetList[nyIndArr[0][ind]] + '\n'
    
print 'positive ny \n\n'
tList = np.nonzero(resNy > np.percentile(resNy, 99))
for ind in tList[0]:
    print tweetList[nyIndArr[0][ind]] + '\n'



print 'negative mc \n\n'
tList = np.nonzero(resMc < np.percentile(resMc, 1))
for ind in tList[0]:
    print tweetList[mcIndArr[0][ind]] + '\n'
    
print 'positive mc \n\n'
tList = np.nonzero(resMc > np.percentile(resMc, 99))
for ind in tList[0]:
    print tweetList[mcIndArr[0][ind]] + '\n'



print 'negative chl \n\n'
tList = np.nonzero(resChl < np.percentile(resChl, 1))
for ind in tList[0]:
    print tweetList[chlIndArr[0][ind]] + '\n'
    
print 'positive chl \n\n'
tList = np.nonzero(resChl > np.percentile(resChl, 99))
for ind in tList[0]:
    print tweetList[chlIndArr[0][ind]] + '\n'



print 'negative kkr \n\n'
tList = np.nonzero(resKkr < np.percentile(resKkr, 1))
for ind in tList[0]:
    print tweetList[kkrIndArr[0][ind]] + '\n'
    
print 'positive kkr \n\n'
tList = np.nonzero(resKkr > np.percentile(resKkr, 99))
for ind in tList[0]:
    print tweetList[kkrIndArr[0][ind]] + '\n'



print 'negative rm \n\n'
tList = np.nonzero(resRm < np.percentile(resRm, 1))
for ind in tList[0]:
    print tweetList[rmIndArr[0][ind]] + '\n'
    
print 'positive rm \n\n'
tList = np.nonzero(resRm > np.percentile(resRm, 99))
for ind in tList[0]:
    print tweetList[rmIndArr[0][ind]] + '\n'



print 'negative lfc \n\n'
tList = np.nonzero(resLfc < np.percentile(resLfc, 1))
for ind in tList[0]:
    print tweetList[lfcIndArr[0][ind]] + '\n'
    
print 'positive lfc \n\n'
tList = np.nonzero(resLfc > np.percentile(resLfc, 99))
for ind in tList[0]:
    print tweetList[lfcIndArr[0][ind]] + '\n'





