import tensorflow as tf
import numpy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Hide the "rebuild TensorFlow with the appropriate compiler flags" warnings

mnist = tf.keras.datasets.mnist

#convert sample data, from int to floating-point numbers
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

x_train, x_test = x_train / 255.0,  x_test / 255.0

#tf.keras.Sequential model by stacking layers
#Sequential provides training and inference features on this model.
# and groups a linear stack of layers into a tf.keras.Model.

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)

])
'''
For each example, the model returns a vector of logits 
or log-odds scores, one for each class.
'''

predictions = model(x_train[:1]).numpy()
print(predictions)

tf.nn.softmax(predictions).numpy()

# define a loss function for training, to take vectors of logits
#and a True index and return a scalar loss for each example

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#this loss will be equal to the log probability of the true class,
#the loss is zero if the model is sure of the correct class.

loss_fn(y_train[:1], predictions).numpy
print("\n", loss_fn(y_train[:1], predictions).numpy)
#This untrained model gives probabilities close to random (1/10 for each class), 
#so the initial loss should be close to -tf.math.log(1/10) ~= 2.3

model.compile(optimizer = "adam",
              loss = loss_fn,
              metrics = ["accuracy"])


model.fit(x_train, y_train, epochs=5)
#Model.fit will adjust the parameters and minimize the loss
#An epoch is a full iteration over samples, 
#epochs represent the nr. of times the algo will run        


model.evaluate(x_test, y_test, verbose=1)
# .evaluate checks the models performance
#verbose=1 will show you an animated progress bar, verbose can be also = 0 or 2


probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print("\n", probability_model(x_test[:5]))
#return the  probability model
