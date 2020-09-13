# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Sets random seed for reproducibility
seed_int = 77
np.random.seed(seed_int)

# Generator class to augment data
class Augmenter:
    def __init__(self, data, batch_size, prob = [1, 0.5]):
        # Data used to generate batches
        self.data = data
        # Size of batch to be generated
        self.batch_size = batch_size
        # Probability of applying any of the augmentation methods to each of the data points
        # Each element of list represents probability applied to each seperate augmentation method
        # Set first element to 1 as a dummy value for applying oversampling (detailed in documentation)
        prob[0] = 1
        self.prob = prob

    def __next__(self):
        # Oversampling method: creates duplicates of data entries. The probability of duplicating an entry is
        # dependant on the 'weights' parameter
        # If probability is not zero
        if self.prob[0]:
            weights = sample_weights(data['Purpose'])
            weights = weights[0:len(self.data)]  # resize to match original self.data
            new_samples = self.data.sample(n = self.batch_size, replace = True, weights = weights)

        # Synonym Replacement method: replaces each word in Description attribute with a randomly selected synonym
        # If probability is not zero
        if self.prob[1]:
            new_samples['Description'] = synonym_replace(new_samples['Description'], self.prob[1])
    
        return new_samples
        
    def __iter__(self):
        return self

# Computes disparity between frequency of each unique value against the most frequent unique value, in data. The
# 'normalize' parameter when True, gives a proportional representation of the disparity
def disparity_count(data, normalize = False):
    counts = data.value_counts()
    disparity = max(counts) - counts
    if normalize:
        return disparity / sum(disparity)
    return disparity

# Computes weights to apply to entries of each class when oversampling
def sample_weights(data):
    # Disparity scaled to the power of 20, arbitrarily, to greatly increase weights assigned to entries of more
    # infrequent classes.
    disparity_scaled = disparity_count(data, True)**20
    weights = []
    for element in data:
        weights = weights + [disparity_scaled[element]]
    return weights

# Computes Boolean decision based on probability
def decide(prob):
    return random.random() < prob

# Concatenates a list of strings together, with spaces
def detokenize(words):
    # Initialise empty string to append words to
    sentence = ""
    for w in words:
        sentence = sentence + " " + w
    return sentence

# Replaces each word in a text dataframe with a corresponding synonym
def synonym_replace(texts, prob):
    # Loops through indices of text dataframe
    for index in range(0, len(texts)-1):
        # Decides whether to continue based on probability parameter
        if decide(prob):
            # Gets single text entry
            entry = texts.iloc[index]
            # Initialises empty synonym list
            syns = []
            # Loops through each word in text
            for w in word_tokenize(entry):
                # Gets potential synonyms through wordnet
                synonyms = wordnet.synsets(w)
                # If synonyms exist for word
                if len(synonyms) > 0:
                    # Selects random synonym
                    rand_lems = random.choice(synonyms)
                    rand_lem = random.choice(rand_lems.lemmas())
                    rand_syn = rand_lem.name()
                else:
                    # Selects original word if no synonyms exist
                    rand_syn = w
                # Appends selected synonym to synonym list
                syns.append(rand_syn.lower())
            # Concatenates synonyms together with spaces, to form realistics Description data point
            texts.iloc[index] = detokenize(syns)         
    return texts

# Creates a frequency bar chart
def plot_freq_bar(data, col_name = 'Purpose'):
    freqs = data[col_name].value_counts()
    freqs = freqs.sort_index()
    return freqs.plot.bar()

# Reads in precprocessed data
data = pd.read_csv('preprocessed_data.csv')
# Removes dud column
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Plots frequency bar chart of class variable
plot_freq_bar(data)
print('Close figure window to continue.')
plt.show()

print("Augmenting data...")

# Initialises augmenter object
Oversample = Augmenter(data, batch_size = 500)
# Generates 25 500-sized batches of data
for i in range(25):
    data = data.append(next(Oversample))

# Plots frequency bar chart of class variable after augmentation
plot_freq_bar(data)
print('Close figure window to continue.')
plt.show()

# Writes augmented data to new csv file
data.to_csv('augmented_data.csv')
