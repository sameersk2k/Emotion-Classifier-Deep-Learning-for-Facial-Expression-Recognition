# Emotion-Classifier-Deep-Learning-for-Facial-Expression-Recognition

## Image Classification Model - Happy or Sad

Overview

This project implements an image classification model using TensorFlow, designed to classify images into two categories: "Happy" and "Sad." The model is built using a Convolutional Neural Network (CNN) architecture and follows a series of steps, including data preprocessing, model creation, training, evaluation, and model saving.

### Table of Contents

- **Dependencies and Setup**
  - Ensure that necessary dependencies, including TensorFlow, are installed. GPU memory growth is configured to avoid out-of-memory errors.
  - This step is crucial for initializing the project environment. The configuration of GPU memory growth is done to prevent potential memory overflow issues during training on compatible hardware.
- **Data Cleaning**
  - Remove images with improper extensions to clean the dataset.
- **Loading Data**
  - Load the image dataset using TensorFlow's image_dataset_from_directory function.
- **Scaling Data**
  - Scaling pixel values to the range [0, 1] is crucial for standardizing input data, enhancing convergence during training, and ensuring consistent behavior across different datasets.
- **Splitting Data**
  - Splitting the dataset into training, validation, and test sets allows for effective model evaluation, preventing overfitting and providing a realistic assessment of generalization.
- **Building Deep Learning Model**
  - **Importing Necessary Modules**:
    
   ```
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout`
   ```
  
    - Sequential: It is a linear stack of layers that allows for the creation of a neural network layer by layer.
    - Conv2D: Convolutional layer for 2D spatial convolution over images.
    - MaxPooling2D: Max pooling operation for spatial data.
    - Dense: Fully connected layer.
    - Flatten: Flattens the input, which is necessary when transitioning from convolutional layers to fully connected layers.
    - Dropout: A regularization technique where randomly selected neurons are ignored during training, reducing overfitting.
      
  - **Model**

    The Sequential model is a linear stack of layers. The layers are added one by one, and it represents a feedforward neural network.

    ```
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    ```
    Adding Convolutional and Pooling Layers:

    Conv2D: Adds a 2D convolutional layer. Here, it adds three convolutional layers with different parameters:
    The first layer has 16 filters of size (3,3), uses a stride of 1, ReLU activation, and expects input of shape (256,256,3).
    The second and third layers follow a similar pattern but with 32 and 16 filters, respectively.

    MaxPooling2D: Adds a 2D max pooling layer. It reduces the spatial dimensions of the input data.

    Flattening Layer:
    Flatten: Converts the multi-dimensional output of the convolutional/pooling layers into a one-dimensional array. This is necessary when transitioning from convolutional layers to fully connected layers.

    Adding a Fully Connected Layer:
    Dense: Adds a fully connected layer with 256 neurons and ReLU activation. This layer processes the flattened output from the previous layer.

    Output Layer:
    Another Dense layer with a single neuron and a sigmoid activation function. This is typical for binary classification problems. The output will be a value between 0 and 1, representing the probability of the input belonging to class 1.

    Compiling the Model:
    - compile: Configures the model for training.
    - 'adam': Adam optimization algorithm, a popular choice for gradient-based optimization algorithms.
    - loss=tf.losses.BinaryCrossentropy(): Binary crossentropy loss function, suitable for binary classification problems.
    - metrics=['accuracy']: During training, monitor accuracy as one of the evaluation metrics.
    ```
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    ```
    
  - **Training the Model**:
    Training the model involves optimizing its weights based on the gradient of the loss function. The training process allows the model to learn patterns and relationships within the training data.

    ```
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
    ```
    Training the Model:

    model.fit: This method trains the model on the specified dataset (train) for a given number of epochs.

    epochs=20: The number of times the model will iterate over the entire training dataset.

    validation_data=val: During training, the model's performance on the validation dataset (val) will be monitored.

    callbacks=[tensorboard_callback]: Callbacks are functions that are called during training. In this case, the TensorBoard callback is used to log information that can be visualized using TensorBoard.

    **Why TensorBoard**?
    - Visualization: TensorBoard provides visualizations of various metrics such as loss and accuracy over time. This helps in understanding how well the model is learning.
    - Model Architecture: You can visualize the architecture of your model using TensorBoard.
    - Embeddings: It allows you to visualize high-dimensional embeddings, which is helpful in understanding how the model is representing data.
    
    By incorporating TensorBoard during training, you can gain valuable insights into the training process and make informed decisions to improve your model.
    
  - Performance Visualization
    Visualizing loss and accuracy during training provides insights into the model's performance. It helps identify potential issues like overfitting or underfitting.
    **Loss**
  <img width="658" alt="Screenshot 2023-11-25 at 11 51 12 PM" src="https://github.com/sameersk2k/Emotion-Classifier-Deep-Learning-for-Facial-Expression-Recognition/assets/115322069/60a27980-44b9-40c5-93ed-5e3702e42938">
    **Accuracy**
  <img width="658" alt="Screenshot 2023-11-25 at 11 51 23 PM" src="https://github.com/sameersk2k/Emotion-Classifier-Deep-Learning-for-Facial-Expression-Recognition/assets/115322069/673abab3-2254-4e0a-99bd-f71ae84cf551">

  - Model Evaluation
    Model evaluation on the test set using precision, recall, and accuracy metrics quantifies the model's performance on previously unseen data.
    **Precision , Recall and Accuracy resulted in 1 which is 100%.**
  - Testing
    Testing the model on new images assesses its ability to make accurate predictions on unseen data, providing practical insights into real performance.

    <img width="665" alt="Screenshot 2023-11-25 at 11 56 27 PM" src="https://github.com/sameersk2k/Emotion-Classifier-Deep-Learning-for-Facial-Expression-Recognition/assets/115322069/dae7f98a-3c8e-4814-b68f-80fd2e380901">

    The output was:
    array([[0.03455482]], dtype=float32). It means it was classified as Happy(closer to Zero).
    
    <img width="428" alt="Screenshot 2023-11-25 at 11 56 46 PM" src="https://github.com/sameersk2k/Emotion-Classifier-Deep-Learning-for-Facial-Expression-Recognition/assets/115322069/017281e4-3259-413a-9c04-52cb36e0bc1b">
    
    The output was:
    array([[0.999627]], dtype=float32) It means it was classified as Sad(CLoser to One).
