import matplotlib
##matplotlib.use('PS')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

########################################################################################
## Author: Rohan Gujral
## Project: NN AI based Germline variant classifier with tensorflow
########################################################################################

########################################################################################
#### Reading training data
#### File name: cleaned_published_data_matrix_training_set
########################################################################################
#### PLEASE PROVIDE PATH TO 'cleaned_published_data_matrix_training_set file

TT_Data = "Path to training data set cleaned_published_data_matrix_training_set"

def read_dataset(filePath):
    df = pd.read_csv(filePath, sep="\t")
    print(df.shape)
    X = df[df.columns[0:23]].values
    y = df[df.columns[23]]

    # Encode th dependent Variables
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    # print(Y)
    print(X.shape)
    return (X, Y)

def read_datasetAlt(filePath):

    df = pd.read_csv(filePath, sep="\t")
    print(df.shape)
    X = df[df.columns[0:23]].values
    Y = df[df.columns[23]].values

    print(X.shape)
    return (X, Y)



# Define the encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# Read the Dataset
X, Y = read_dataset(TT_Data)

# Shuffle the dataset to mix rows because in beggining of dataset there are only mines and then rocks...so Shuffle.
##X, Y = shuffle(X, Y, random_state=1)

# Split the dataset into train and test
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# Check or quick look at the train and test shape
print("\nTrain and Test Shape")
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

# Define important parameters and variables to work with tensors
learning_rate = 0.1
training_epochs = 300  # Total Number of iterations will be done to minimize the error
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim = ", n_dim)
n_class = 2

############################################################
#### Path for storing the model 
#### Please change location if you want to save model.
###########################################################
#### PLEASE CHANGE THE model_path
#model_path = "/Users/rgujral/software-projects/PycharmProjects/tensorflow/data/forestDNMModel"  # Path to store the model

### MULTILAYER PERCEPTRON

# Define number of hidden layers and number of neurons for each layer
n_hidden_1 = 100
n_hidden_2 = 100
n_hidden_3 = 80
n_hidden_4 = 80
n_hidden_5 = 80
n_hidden_6 = 80
n_hidden_7 = 60
n_hidden_8 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])  # Output of our model


# Define the model
def multilayer_perceptron(x, weights, biases):
    # Hidden Layer 1 with Sigmoid Activation function
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden Layer 2 with Sigmoid Activation function
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden Layer 3 with Sigmoid Activation function
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden Layer 4 with Relu Activation function
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    ## layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.nn.sigmoid(layer_4)

    # Hidden Layer 5 with Sigmoid Activation function
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.sigmoid(layer_5)

    # Hidden Layer 6 with Sigmoid Activation function
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.sigmoid(layer_6)

    # Hidden Layer 7 with Sigmoid Activation function
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.sigmoid(layer_7)

    # Hidden Layer 6 with Relu Activation function
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf.nn.relu(layer_8)

    # Output layer with linear activation
    # out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
    return out_layer


# Define the weights and biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),  # 60 x 60
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.truncated_normal([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(tf.truncated_normal([n_hidden_7, n_hidden_8])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_8, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'b5': tf.Variable(tf.truncated_normal([n_hidden_5])),
    'b6': tf.Variable(tf.truncated_normal([n_hidden_6])),
    'b7': tf.Variable(tf.truncated_normal([n_hidden_7])),
    'b8': tf.Variable(tf.truncated_normal([n_hidden_8])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# Initialize all the variables
init = tf.global_variables_initializer()

# Saver object in order to save the model
saver = tf.train.Saver()

# Call the model Defined above
y = multilayer_perceptron(x, weights, biases)

# Define the cost/loss function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Tensoflow Session
sess = tf.Session()
sess.run(init)

# Calculte the cost and accuracy of each epoch
mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_, 1))  # Difference between the actual output and model outputs
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))  # Mean square error
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)

    print("Epoch : ", epoch, " - Cost:", cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

###save_path = saver.save(sess, model_path) #Uncomment this line if you wan to save path
###print("Model saved in file: %s" % save_path)

# Plot mse and accuracy grapth
plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history, 'b')
plt.show()

# Print the final accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

# Print the final mean square error
pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))


##############################################################
####### Making predictions ###################################
########## on test data ######################################

totalPosCount = 0
incorrectPosCount = 0
correctPosCount = 0

totalNegCount = 0
incorrectNegCount = 0
correctNegCount = 0

prediction=tf.argmax(y,1)
prediction_run = sess.run(prediction, feed_dict={x: test_x})


for i in range(1, len(prediction_run)):
    result = prediction_run[i]
    if(result == 1):
        totalPosCount = totalPosCount + 1
        if(test_y[i][1] == 1):
            correctPosCount =  correctPosCount  + 1
            print("" + str(prediction_run[i]) + "\t" + str(test_y[i]) + "\t" + str(test_x[i]))
        else:
            incorrectPosCount = incorrectPosCount + 1

    else:
        totalNegCount = totalNegCount + 1
        if (test_y[i][1] == 0):
            correctNegCount = correctNegCount + 1
            ##print("" + str(prediction_run[i]) + "\t" + str(test_y[i]) + "\t" + str(test_x[i]))
        else:
            incorrectNegCount = incorrectNegCount + 1

print("Total pos Count: ", totalPosCount)
print("Incorrect pos count: ", incorrectPosCount)
print("Correct  pos count: ", correctPosCount)

print("Total neg Count: ", totalNegCount)
print("Incorrect neg count: ", incorrectNegCount)
print("Correct  neg count: ", correctNegCount)


"""
for i in range(1, len(X)):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 23)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 23), y_: Y[i].reshape(1, 2)})
    if(prediction_run[0] == 1):
        print(i, "Original Class: ", int(sess.run(y_[i][1], feed_dict={y_: Y})), " Predicted Values: ", prediction_run[0])
        print(i, "Original ClassA: ", int(sess.run(y_[i][0], feed_dict={y_: Y})), " Predicted Values: ", prediction_run[0])
    ##print("Accuracy: ", str(accuracy_run * 100) + "%")
"""

##############################################################
####### Making predictions ###################################
########## on actual data ####################################

#################################
#### Path to validation data ####
#### Small validation data   ####
#### set is provided here    ####
#### for evaluation.         ####
#### smallValidationData     ####
#################################

predData = "Path to sample validation data smallValidationData"

Xpred, Ypred = read_datasetAlt(predData)
pred_run = sess.run(prediction, feed_dict={x: Xpred})

probabilities = sess.run(tf.nn.softmax(y),feed_dict={x: Xpred})

##pred_run = sess.run(predict, feed_dict={x: Xpred})

counterOnRealData = 0

for i in range(1, len(pred_run)):
    resultPred = pred_run[i]
    if(resultPred == 1):
        counterOnRealData = counterOnRealData + 1
        print("Real data prediction: " + str(resultPred) + "\t" + str(Xpred[i]) +  "\t" + str(Ypred[i]) +  "\t" + str(probabilities[i]))


print("Germline calls on real data: " + str(counterOnRealData))

print("Probabilities: " + str(probabilities))
