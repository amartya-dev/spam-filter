"""We use this script in order to pre process the data before feeding it S
into the classifier to predict the result.
We use the following method : the frequency of the words used is 
recorded and returned and then those frequencies are fed into the classifier
to predict weather the given text would be spam or not.
We pre process the input data by :
    1. Removing unnecessary symbols like \n and all
    2. Removing the stopwords and extracting the keywords
"""

#importing all the dependencies
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import os

#================================================================================================
# load stopwords
stopwords = set(stopwords.words('english'))

# To clean the data i.e. remove the stopwords etc. and to get the keywords
def preprocess(message):
    
    # remove '=' symbols before tokenizing, since these
    # sometimes occur within words to indicate, e.g., line-wrapping
    # also remove newlines    
    all_words = set(wordpunct_tokenize(message.replace('=\\n', '').lower()))
    
    # remove the stopwords
    msg_words = [word for word in all_words if word not in stopwords and len(word) > 2]
    return (msg_words)


def get_features(messages):

    """
    Function which converts the
    messages into feature or vectorize form
    """
    vectorizer = CountVectorizer(min_df=1)
    X_count_vec = vectorizer.fit_transform(messages)
    transformer = TfidfTransformer(smooth_idf=False)
    X = transformer.fit_transform(X_count_vec)
    return X
