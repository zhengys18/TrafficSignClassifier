# Load pickled data
import pickle

training_file = 'dataset/train.p'
validation_file= 'dataset/valid.p'
testing_file = 'dataset/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

#########################################################################################
import numpy as np

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# Shape of an traffic sign image
image_shape = X_train.shape[1:]

# Number of unique classes/labels there are in the dataset
n_classes = len(np.unique(np.concatenate((y_train,y_valid, y_test))))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#########################################################################################
import random
import matplotlib.pyplot as plt
#%matplotlib inline

index = random.randint(0, len(X_train))
print(index)
image = X_train[index].squeeze()

plt.figure()
plt.imshow(image)
print(y_train[index])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))
ax1.hist(y_train, n_classes, normed=1)
ax1.set_title('training label distribution')
ax2.hist(y_valid, n_classes, normed=1)
ax2.set_title('validation label distribution')
ax3.hist(y_test, n_classes, normed=1)
ax3.set_title('test label distribution')

#########################################################################################
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
from sklearn.utils import shuffle
from skimage import exposure
import cv2

def preprocess_data(X, y = None):
    # Convert to a single Y channel and apply histogram equalization to brighten the image
    X_proc = np.zeros_like(X[:, :, :, 0])
    for i in range(X.shape[0]):
        img_to_yuv = cv2.cvtColor(X[i], cv2.COLOR_RGB2YUV)
        X_proc[i] = cv2.equalizeHist(img_to_yuv[:,:,0])    
    print(X_proc.shape)

    # Normalize image to be between [-1, 1]
    X_proc = ((X_proc-128.0) / 128.0).astype(np.float32)  
    
    # Add dimension for CNN input requirement
    X_proc = X_proc[:,:,:,np.newaxis]

    return X_proc

X_train_proc = preprocess_data(X_train, y = None)
X_valid_proc = preprocess_data(X_valid, y = None)
X_test_proc = preprocess_data(X_test, y = None)

plt.figure()
plt.imshow(X_train[index].squeeze())
plt.figure()
plt.imshow(X_train_proc[index].squeeze(),cmap='gray')

#########################################################################################
import tensorflow as tf

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())


EPOCHS = 51
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def conv2d(input, W, b, paddingType='SAME'):
    return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding=paddingType) + b

def maxpooling(input, size):
    return tf.nn.max_pool(input, 
                           ksize = [1, size, size, 1], 
                           strides = [1, size, size, 1], 
                           padding='VALID')

prob = tf.placeholder_with_default(1.0, shape=())
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    weights = {
        'conv1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma)),
        'conv2': tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = mu, stddev = sigma)),
        'conv3': tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 256), mean = mu, stddev = sigma)),
        'fc1': tf.Variable(tf.truncated_normal(shape=(5632, 1024), mean = mu, stddev = sigma)),
        'fc2': tf.Variable(tf.truncated_normal(shape=(1024, 256), mean = mu, stddev = sigma)),
        'out': tf.Variable(tf.truncated_normal(shape=(256, n_classes), mean = mu, stddev = sigma))}

    biases = {
        'conv1': tf.Variable(tf.zeros(32)),
        'conv2': tf.Variable(tf.zeros(64)),
        'conv3': tf.Variable(tf.zeros(256)),
        'fc1': tf.Variable(tf.zeros(1024)),
        'fc2': tf.Variable(tf.zeros(256)),
        'out': tf.Variable(tf.zeros(n_classes))}  
    
    # Layer 1: Convolutional 5x5. Input = 32x32x1. Output = 32x32x32.
    conv_layer1 = conv2d(x, weights['conv1'], biases['conv1'])
    relu_layer1 = tf.nn.relu(conv_layer1)
    # Pooling. Input = 32x32x32. Output = 16x16x32.
    pooling_layer1 = maxpooling(relu_layer1, 2)
    
    # Layer 2: Convolutional 3x3. Input = 16x16x32. Output = 16x16x64.
    conv_layer2 = conv2d(pooling_layer1, weights['conv2'], biases['conv2'])
    relu_layer2 = tf.nn.relu(conv_layer2)
    # Pooling. Input = 16x16x64. Output = 8x8x64.
    pooling_layer2 = maxpooling(relu_layer2, 2)
    
    # Layer 3: Convolutional 3x3. Input = 8x8x64. Output = 8x8x256.
    conv_layer3 = conv2d(pooling_layer2, weights['conv3'], biases['conv3'])
    relu_layer3 = tf.nn.relu(conv_layer3)
    # Pooling. Input = 8x8x256. Output = 4x4x256.
    pooling_layer3 = maxpooling(relu_layer3, 2)
    
    # Combine branch 1,2 together and Flatten. Input = 4x4x32 + 4x4x64 + 4x4x256. Output = 5632
    fc_layer1 = tf.concat([maxpooling(pooling_layer1, 4), maxpooling(pooling_layer2, 2), pooling_layer3], 3)
    fc_layer1 = flatten(fc_layer1)
    dropout_layer1 = tf.nn.dropout(fc_layer1, prob)
    
    # Layer 4: Fully Connected. Input = 5632. Output = 1024.
    fc_layer2 = tf.add(tf.matmul(dropout_layer1, weights['fc1']), biases['fc1'])
    relu_layer2 = tf.nn.relu(fc_layer2)
    dropout_layer2 = tf.nn.dropout(relu_layer2, prob)
    
    # Layer 5: Fully Connected. Input = 1024. Output = 256.
    fc_layer3 = tf.add(tf.matmul(dropout_layer2, weights['fc2']), biases['fc2'])
    relu_layer3 = tf.nn.relu(fc_layer3)
    dropout_layer3 = tf.nn.dropout(relu_layer3, prob)

    # Layer 5: Fully Connected. Input = 256. Output = n_classes.
    fc_layer4 = tf.add(tf.matmul(dropout_layer3, weights['out']), biases['out'])
    logits = tf.nn.relu(fc_layer4)
    
    return logits

#########################################################################################
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/gpu:0'):
    logits = LeNet(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

# Learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10, 0.9, staircase=True)
# Passing global_step to minimize() will increment it at each step.
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    total_accuracy = 0
    sess = tf.get_default_session()
    num_examples = len(X_data)
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

training_history = []

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_proc, y_train = shuffle(X_train_proc, y_train)
        for offset in range(0, n_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_proc[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, prob: 0.7})
            
        validation_accuracy = evaluate(X_valid_proc, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        training_history.append([i+1, validation_accuracy])

        if (i % 10 == 0):
            saver.save(sess, './lenet')
            print("Model saved")

