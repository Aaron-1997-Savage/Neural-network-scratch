# Neural-network-scratch
Some scripts for a simple neural network

There are 2 scripts in this file.

>#### **1. Network.py**: 
***Contains several functions and classes, including ReLU, softmax, crossentropy and other utility functions. Additionally, the feed forward and back propagation algorithm are implemented in the script.***

***(1) class nn:***

***init_parameters***: 

For initializing parameters.
"n_hidden" is the structure of the network, "activation list" is the activation function of each hidden layer.

***output***: 

It is quite similar to a single layer of feed forward network, which outputs the value (or matrix) after function operation.
output function will output "z", which is "weights" dot "inputs" plus "bias", "a" is the activation function output of "z". This function could be regarded as a single hidden layer.

***(2) class Model: super from 'nn'***
        
***forward_pass***: 

By passing through the parameters, outputs of every hidden layer are solved
concatenate all the outputs of each hidden layer, which return from the function "outputs". More details about the feed forward pass network, please look at page 5 of the slides "scratch.pptx"

![image](https://github.com/Andrewhuang723/Neural-network-scratch/blob/main/forward_pass.png)

***backward***: 

a simple algorithm of back propagation
dcda_2 is the derivative of the loss to the output of the current hidden layer, z is the current pre-activation value, activation_function is the activation function of current layer"a_1" is the output of the previous hidden layer
        
***back_propagation***:

"n_hidden", "activation_list" is the concatenate of hidden layers, activation functions. "pred" is the output of the forward_pass, "expected" is the ground-truth
"parameters" is the concatenate of all the weights and bias in different activation function of hidden layer. Return the gradients of loss.

![image](https://github.com/Andrewhuang723/Neural-network-scratch/blob/main/back_propagation.png)

***update***:

W := W - (learning rate) * gradients
alpha = learining rate

***(3) mini_batch***:

Slicing the data into n_batches, each batch contains "batch_size" X and Y.
    
***(4) get_acc***:

According to the probability distribution based on softmax, if the index in prediction has the maximum probability, and matches the maximum value in ground truth data, then it is a correct prediction.
the function returns the percentage in which the correct prediction labels over total predictions.
y is ground truth and y_hat is prediction.
    
***(5) mini_batch_training***:

Slicing data into batches for training by "mini_batch". Get the initial parameters from data, "X_train", "y_train" and the model structure, "n_hidden", "activation_list".
For each epoch in "epochs", each batch of data will passing through the feed forward network, class "Model().forward_pass()", returns predictions. Passing backward by class "Model().backpropagation()",
and updated by gradient descent from class "Model().update()", the updated parameters is solved, iteratively, the terminal parameters are saved in the dictionary "model_para".
Additionally, the loss and accuracy of each epoch are saved in the dictionary, "history". Same process with validation data, "X_val", "y_val".

    model, history = mini_batch_training(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, n_hidden=structure, activation_list=act_list, batch_size=128, epochs=100)
    
    0/329 loss: 0.25318325  Accuracy: 0.1172
    1/329 loss: 0.25337678  Accuracy: 0.1250
    2/329 loss: 0.25407721  Accuracy: 0.0391
    3/329 loss: 0.25344646  Accuracy: 0.0625
    4/329 loss: 0.25340568  Accuracy: 0.0938
    
    
***(6) predict***:

Predict the testing data "X_test", "y_test" with the optimized parameter "model_para" by "mini_batch_training", outputs the prediction, loss, and accuracy: "Y_pred", "loss", "accuracy".

    y_pred, loss, acc = predict(X_test=X_test, y_test=y_test, model_para=model, n_hidden=structure, activation_list=act_list)

    

>#### **2. Model.py**: contains reading data and data preprocessing, and last, the model training as well as prediction is carried out.

***(1) num_images***:
    
Returns the number of images, which are saved in zip file

***(2) read_image***:
    
Read images from the file, the images are 784 pixels, by slicing the data in 784 per index and reshape to (28, 28), returns the array of the images.
    
***(3) read_label***:
    
Read labels from the file, returns the array with shape (n, 1)
    
***(4) one_hot_encode***:
    
Presenting label data as one hot encoded label.
    
***(5) shuffle***:
    
Make the data more flexible by shuffling the indices of data.

***(6) split***:
    
Splitting training data into training data and validation data.

>#### **3. Visualizing **

Here I choose t-distributed stochastic neighbor embedding (t-SNE) for visualizing the data.
Like PCA, t-SNE is an useful method for dimension reduction, but also solve the problem of curse of dimensionality and crowding problem in PCA.
More details about t-SNE, [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) is recommended reading.

    t_sne = TSNE(n_components=2, perplexity=30)
    t_sne_y_val = reversed_one_hot(y_val)
    t_sne_val = t_sne.fit_transform(X_val)
    val_data = np.concatenate((t_sne_val, t_sne_y_val), axis=1)
    val_data = pd.DataFrame(val_data, columns=["feature_1", "feature_2", "label"])
    sns.FacetGrid(val_data, hue="label", size=6).map(plt.scatter, "feature_1", "feature_2", s=5).add_legend()

![image](https://github.com/Andrewhuang723/Neural-network-scratch/blob/main/test_tsne.png)
