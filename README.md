# Monk-AI-Assignment

Implement a 3-class classification neural network with a single hidden layer using Numpy.This model is a class of artificial neural network that uses back propagation technique for training. 

Algorithm 
-
Import Libraries - We will import some basic python libraries like numpy, matplotlib (for plotting graphs), sklearn (for data mining and analysis tool), etc. that we will need.

Dataset - We will use the Iris Dataset, It contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). It is a multi class classification problem. Now, let us divide the data into a training set and test set. This can be accomplished using sklearn train_test_split() function. 20% of data is selected for test and 80% for train. Also, we will check the size of the training set and test set. This will be useful later to design our neural network model.

Neural Network Model - 

The general methodology to build a Neural Network is to:
1. Define the neural network structure ( # of input units,  # of hidden units, etc).

2. Initialize the model's parameters. 

3. Loop:
    - Implement forward propagation
    - Compute loss
    - Implement backward propagation to get the gradients
    - Update parameters (gradient descent). 

Define structure - We need to define the number of input units, the number of hidden units, and the output layer. The input units are equal to the number of features in the dataset (4), hidden layer is set to 4 (for this purpose), and the problem is the 3 class classification we will use a 3 neuron in output.

Initialize Model Parameter - We need to initialize the weight matrices and bias vectors. Weight is initialized randomly while bias is set to zeros.Also we need to initilize activation fumction parameter K0 & K1. we initilize them randomly with (K0 = np.random.randint(10)*0.1) & (K1 = np.random.randint(10)*0.1) these two function.

Forward Propagation - For forward propagation, given the set of input features (X), we need to compute the activation function for each layer. For the hidden layer, we are using (activation function: g(x) = k0 + k1x), Similarly, for the output layer, we are using (softmax activation function: g(x) = Softmax(x)).

Compute Loss - We will compute the cross-entropy cost. In the above section we calculated the output of our output layer. Using this output we can compute cross-entropy loss.

Backpropagation - We need to calculate the gradient with respect to different parameters such as dK0, dK1, dW1, dB1, dW2, dB2.

Gradient Descent (update parameters) - We need to update the parameters using the gradient descent rule i.e. theta = theta - alpha(dj/d_theta).

Neural Network Model - Finally, putting together all the functions we can build a neural network model with a single hidden layer.

Prediction  - Using the learned parameter, we can predict the class for each example by using forward propagation. we get the training accuracy is around 99.19% which means that our model is working and fit the training data with high probability. The test accuracy is around 93.33%. Training F1 score is 0.99 and test F1 score is 0.99.Given the simple model and the small dataset, we can consider it as a good model.

We have also generated view in our python code. view - Parameter Update on epoch & Loss function Vs Epoch.






