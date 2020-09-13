# Load in libraries
import pandas as pd
import numpy as np
# Libraries for NLP
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
# Libraries for classification
from sklearn import model_selection, svm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
# Library for confusion matrix
import seaborn as sn
import matplotlib.pyplot as plt 

# Sets random seed for reproducibility
seed_int = 77
np.random.seed(seed_int)

# Reads in augmented data
data = pd.read_csv('augmented_data.csv')

# Selects Description field and class variable
nlp_data = data.loc[:, ['Description', 'Purpose']]

print("Natural Language Processing...")

# Tokenisation: each entry in nlp_data will be broken into a set of words
nlp_data['Description'] = [word_tokenize(entry) for entry in nlp_data['Description']]

# Remove stop words, non-alphabetical words and perform word lemmatizing/stemming.
# WordNetLemmatizer requires POS (Parts Of Speech) tags to understand if the word is a noun, verb, adverb or
# adjective.
# The object 'defaultdict' is a dictionary that automatically creates new entries for undefined keys by giving
# it a default value. Here, the default value is Noun.
tag_map = defaultdict(lambda : wordnet.NOUN)
tag_map['J'] = wordnet.ADJ
tag_map['V'] = wordnet.VERB
tag_map['R'] = wordnet.ADV

# Method 'enumerate' adds a counter to an iterable and returns it as an enumerate object
for index, entry in enumerate(nlp_data['Description']):
    # Initialises empty list to store the processed words of the description
    processed_wordset = []
    # Initialises WordNetLemmatizer()
    word_lemmatized = WordNetLemmatizer()
    # pos_tag function returns the 'tag' of a word
    for word, tag in pos_tag(entry):
        # Condition checks for stop words and considers only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            processed_word = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            processed_wordset.append(processed_word)
    # The final processed description for each entry will be stored in 'processed_description'
    nlp_data.loc[index, 'processed_description'] = str(processed_wordset)

print("Splitting...")

# Splits data into training and test sets, and each further into feature (X) and target (Y) sets
test_proportion = 0.25
training_X, test_X, training_Y, test_Y = model_selection.train_test_split(nlp_data['processed_description'],
                                                            nlp_data['Purpose'], test_size = test_proportion)

print("TFIDF Processing...")

# Term Frequency-Inverse Document Frequency (TFIDF) Processing. TFIDF is a weighting applied more heavily to words
# that appear more frequently in a text and is offset by their frequency within the set of texts.
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(nlp_data['processed_description'])
training_X_tfidf = tfidf_vect.transform(training_X)
test_X_tfidf = tfidf_vect.transform(test_X)

print('Baseline modelling...')

# Initialise strings of baseline strategies to use
baseline_strategies = ['stratified', 'most_frequent', 'uniform']
# Model each baseline strategy
for strat in baseline_strategies:
    dummy = DummyClassifier(strategy = strat)
    dummy_train = dummy.fit(training_X, training_Y)
    dummy_train_score = dummy_train.score(training_X, training_Y)
    dummy_test = dummy.fit(test_X, test_Y)
    dummy_test_score = dummy_test.score(test_X, test_Y)
    print('Baseline', strat, 'model scored ', round(dummy_train_score*100, 1), '% on training data')
    print('Baseline', strat, 'model scored ', round(dummy_test_score*100, 1), '% on test data')

print('Fitting...')

# Initialises SVM classifier, 'probability = True' to use to ensemble later 
SVM = svm.SVC(C = 1, kernel = 'linear', probability = True)
# Fits training data with SVM
SVM_model = SVM.fit(training_X_tfidf, training_Y)
# Calculate accuracy score of classifier
# Accuracy is calculated by (TP + TN) / (TP + TN + FP + FN)
SVM_score = SVM_model.score(test_X_tfidf, test_Y)
print('SVM accuracy score =', SVM_score*100)

# Initialises, fits and scores Logistic Regression classifier
LR = LogisticRegression(random_state = seed_int)
LR_model = LR.fit(training_X_tfidf, training_Y)
LR_score = LR_model.score(test_X_tfidf, test_Y)
print('Logistic Regression accuracy score =', LR_score*100)

print("Ensembling...")

# Ensembles two classifiers. Parameter 'voting = 'soft'' allows class labels to be predicted based on the
# prediction probabilities of each classifier.
ensemble = VotingClassifier(estimators = [('SVM', SVM_model), ('LR', LR_model)], voting = 'soft', weights = [1, 1])
ensemble_model = ensemble.fit(training_X_tfidf, training_Y)
ensemble_score = ensemble_model.score(test_X_tfidf, test_Y)
print('Ensemble accuracy score =', ensemble_score*100)

print("Computing confusion matrix...")

# Creates confusion matrix, using predicted labels from ensemble classifier
ensemble_pred = ensemble.predict(test_X_tfidf)
ensemble_pred = ensemble_pred.astype(int) # for readability
test_Y = test_Y.astype(int)
conf_mat = pd.crosstab(test_Y, ensemble_pred, rownames = ['Actual'], colnames = ['Predicted'])

# Plots confusion matrix
plotted_mat = sn.heatmap(conf_mat, annot = True, cmap = 'Greens', fmt = 'd')
plotted_mat.set_ylim(13, 0)
print('Close figure window to finish.')
plt.show()

# Creates normalised confusion matrix
conf_norm = conf_mat.div(conf_mat.sum(axis = 1), axis = 0)
# Ensures classes with no predictions will be displayed with 0s instead of NaNs
conf_norm = conf_norm.replace(np.nan, 0.00)
conf_norm = conf_norm.round(2)


