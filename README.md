# coursera_machine_learning
Coursework for the Introduction to Machine Learning course by Andrew Ng on Coursera.

*Exercise 1:*
In this part of this exercise, you will implement linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next.

- plotted the data
- implemented gradient descent
- implemented linear regression


*Exercise 2:*
Build a logistic regression model to predict whether a student gets admitted into a university. Suppose that you are the administrator of a university department and you want to determine each applicant's chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant's scores on two exams and the admissions decision.

- plotted data
- implemented the sigmoid function
- implemented the cost function and gradient descent 
- implemented regularized logistic regression
- evaluated logistic regression model


*Exercise 3:*
For this exercise, you will use logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. This exercise will show you how the methods you've learned can be used for this classication task. In the first part of the exercise, you will extend your previous implemention of logistic regression and apply it to one-vs-all classification.

- vectorized the cost function, gradient descent, and logistic regression for faster computing
- implemented one-vs-all multiclass logistic regression to recognize handwritten digits
- implemented a neural network to recognize handwritten digits


*Exercise 4:*
In the previous exercise, you implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights we provided. In this exercise, you will implement the backpropagation algorithm to learn the parameters for the neural network.

- implemented feedforward propogation
- implemented regularized cost function and gradient descent
- implemented backpropagation algorithm with random initialization and gradient checking to learn neural network parameters
- implemented regularized neural network to recognize handwritten digits


*Exercise 5:*
In this exercise, you will implement regularized linear regression and use it to study models with different bias-variance properties. In the first half of the exercise, you will implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir. In the next half, you will go through some diagnostics of debugging learning algorithms and examine the effects of bias vs. variance.

- implemented polynomial linear regression
- implemented cross validation to select lambda for regularization


*Exercise 6:*
In this exercise, you will be using support vector machines (SVMs) to build a spam classifier. In the first half of this exercise, you will be using support vector machines (SVMs) with various example 2D datasets. Experimenting with these datasets will help you gain an intuition of how SVMs work and how to use a Gaussian kernel with SVMs. In the next half of the exercise, you will be using support vector machines to build a spam classifier.

- implemented SVMs with Gaussian kernels on datasets that are not linearly separable
- trained SVMs to build a spam filter

*Exercise 7:*
In this exercise, you will implement the K-means clustering algorithm and apply it to compress an image by reducing the number of colors that occur in an image to only those that are most common in that image. In the second part, you will use principal component analysis to find a low-dimensional representation of face images. 

In a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding. Our image contains thousands of colors, and in this part of the exercise, you will reduce the number of colors to 16 colors. By making this reduction, it is possible to represent (compress) the photo in an efficient way. Specifically, you only need to store the RGB values of the 16 selected colors, and for each pixel in the image you now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities). 

- **Implemented K-means clustering to compress an image.** Reduced the number of bits per pixel location from 24 to 4, by storing the representation of 16 main colors in a dictionary and representating the 16 colors using 4 bits per pixel location when compressing the image. Achieved a compression by a factor of 6 - from 128\*128\*24=393,216 bits to 24\*16 +128\*128\*4=65,920. 

<img width="618" alt="Screen Shot 2021-09-20 at 11 31 55" src="https://user-images.githubusercontent.com/30210990/134039392-53c38dd1-7e57-4e40-88c6-347d654af45f.png">

- **Implemented PCA to find a low-dimensional representation of face images.** Used PCA on black and white images (grayscale) of faces and reduced dimensionality of image representation from 1,024 to 100 - some fine details were lost, but the main face characteristics were preserved. 

<img width="739" alt="Screen Shot 2021-09-20 at 11 31 31" src="https://user-images.githubusercontent.com/30210990/134039320-c116b817-1a83-41ac-a648-040c4b00e069.png">



*Exercise 8:*
In this exercise, you will implement the anomaly detection algorithm and apply it to detect failing servers on a network. In the second part, you will use collaborative filtering to build a recommender system for movies.

- implemented anomaly detection
- implemented a recommender system for movies

