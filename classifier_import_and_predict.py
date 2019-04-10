""""We use this script to load the pre trained model 
selected by their scores in the classifier.py file
The pre trained model is then fed the data to predict
by the gui and then this script calls the preprocessing
nethods declared in pre_process.py in order to make the data
ready for prediction"""


#importing all the dependencies
import sklearn
from sklearn.externals import joblib
from pre_process import preprocess,get_features
import numpy as np

#========================================================================================

def predict(message) :
    # file containing the classifier
    filename = 'finalized_model.sav'
    loaded_model = joblib.load(filename)

    # message entered, to be pre processed
    message = preprocess(message)
    y = get_features(message)
    y = y.toarray()

    #Knowing the number of features in the present converted /pre-processed data
    tup = y.shape
    k = tup[1]-1

    # While loop to make the number of features as SVM and adabooster can only work 
    #after association 
    
    if (k > 56) :
        while (k != 56) :   
            y = np.delete(y,k,1)
            k -= 1
    else :
        return("Text too small to obtain features therefore not spam")

    result_array = loaded_model.predict(y)
    if (result_array[-1] == 0) :
        return("Not Spam")
    else :
        return("Spam")

