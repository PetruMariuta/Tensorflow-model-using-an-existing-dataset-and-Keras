# Tensorflow-model-using-an-existing-dataset-and-Keras

This program: 1st loads a prebuilt dataset (MNIST), 2nd Builds a neural network machine learning model that classifies images, 3rd trains a machine learning model using the MNIST database of handwritten digits, using the Keras API and Evaluates the accuracy of the model by returning the probability model

- the MNIST dataset is loaded and then prepared, converted to the sample data from integers to floating-point numbers

-  tf.keras.Sequential builds the model by stacking layers

- converts the logits to probabilities for each class using  tf.nn.softmax(predictions).numpy()

- losses.SparseCategoricalCrossentropy takes a vector of logits and a True bool value and returns the scalar loss

-log-odds, where p is the ratio of the probability of success ratio of the probability of success

![image](https://user-images.githubusercontent.com/118382269/206126996-5680c781-016e-4380-a22a-bf06b7cfda8d.png)

-the untrained model gives probabilities close to random (1/10 for each class)

-before the training, Keras Model.compile  helps to configure and compile the model

- adjust  model parameters and minimize the loss by using Model.fit()

-training is complete, now by using Softmax(), the probability model of the first 5 test data

And the output of the program:
![image](https://user-images.githubusercontent.com/118382269/206130577-8afcd59a-19b3-4828-9e76-5957688cbb60.png)

