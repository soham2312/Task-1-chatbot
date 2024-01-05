import pickle
import numpy as np
import json
import random
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load the Q/A file
QandA_file = open('Question_and_Answers.json').read()
QandA = json.loads(QandA_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!','.',',']

# Loop through each sentence in the Question_and_Answers patterns
for question in QandA['questions_and_answers']:
    for pattern in question['patterns']:
        # Tokenize each word in the sentence
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents in the corpus
        documents.append((word, question['tag']))
        # Add to our classes list
        if question['tag'] not in classes:
            classes.append(question['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# dump words and classes to pickle files
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Create training and testing data
training = []
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence
for document in documents:
    bag = []
    # List of tokenized words for the pattern
    word_patterns = document[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag+output_row)

# Shuffle features and turn into np.array
random.shuffle(training)
training = np.array(training)

# Create train and test lists. X - patterns, Y - intents
train_x = training[:,:len(words)]
train_y = training[:,len(words):]
print(train_x.shape)
# Build model
model=tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]),activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd=tf.keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)

# Save the model
model.save('chatbot_model.h5',hist)

print("Done")

