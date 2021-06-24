# Monk-AI-Assignment
# Name - Rushikesh Bhosale

# Before running the code, download the iris.csv dataset and in your IDE please provide the path of local directory where you have stored the downloaded iris.csv data set

Implement a 3-class classification neural network with a single hidden layer using Numpy.This model is a class of artificial neural network that uses back propagation technique for training. 

Algorithm 
-
Import Libraries - We will import some basic python libraries like numpy, matplotlib (for plotting graphs), sklearn (for data mining and analysis tool), etc. that we will need.

Dataset - We will use the Iris Dataset, It contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). It is a multi-class classification problem. Now, let us divide the data into a training set and test set. This can be accomplished using sklearn train_test_split() function. 20% of data is selected for test and 80% for train. Also, we will check the size of the training set and test set. This will be useful later to design our neural network model.

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

Initialize Model Parameter - We need to initialize the weight matrices and bias vectors. Weight is initialized randomly while bias is set to zeros. Also we need to initialize activation function parameter K0 & K1. we initilize them randomly with (K0 = np.random.randint(10)*0.1) & (K1 = np.random.randint(10)*0.1) these two function.

Forward Propagation - For forward propagation, given the set of input features (X), we need to compute the activation function for each layer. For the hidden layer, we are using (activation function: g1(x) = k0 + k1x), Similarly, for the output layer, we are using (softmax activation function: g2(x) = Softmax(x)).

z1 = a0 x w1 + b1

a1 = g1(z1)

z2 = a1 x w2 + b2

a2 = g2(z2)

Compute Loss - We will compute the cross-entropy cost. In the above section we calculated the output of our output layer. Using this output we can compute cross-entropy loss.

![cross_entropy_loss](https://user-images.githubusercontent.com/55809031/123210081-20526b80-d4df-11eb-9fed-6536e289f2fe.png)

Backpropagation - We need to calculate the gradient with respect to different parameters such as dK0, dK1, dW1, dB1, dW2, dB2.
dZ2 = a2 -  y

dW2 = (a1.T X dZ2) / t

dBw = avgcol(dZ2)

da1 = dZ2 X W2.T

dZ1 = g'(Z1) * da1

dW1 = a0.T X dZ1 

dB1 = avgcol(dZ1)

K0 = avg(da1)

K1 = avg(da1 * z1)

Gradient Descent (update parameters) - We need to update the parameters using the gradient descent rule i.e. theta = theta - alpha(dj/d_theta).

Neural Network Model - Finally, putting together all the functions we can build a neural network model with a single hidden layer.

Prediction  - Using the learned parameter, we can predict the class for each example by using forward propagation. we get the training accuracy is around 99.19% which means that our model is working and fit the training data with high probability. The test accuracy is around 93.33%. Training F1 score is 0.99 and test F1 score is 0.99.Given the simple model and the small dataset, we can consider it as a good model.

Accuracy Train: 99.166667

F1 Score Train: 0.991667

Accuracy Test: 93.333333

F1 Score Test: 0.991667

We have also generated view in our python code. view - Parameter Update on epoch & Loss function Vs Epoch.

![Figure_1](https://user-images.githubusercontent.com/55809031/123169712-43592d00-d497-11eb-837d-35547995e2da.png)

![Figure_2](https://user-images.githubusercontent.com/55809031/123169809-64218280-d497-11eb-9884-774d8a550cb4.png)






