# coursera_machine_learning
Coursework for the Introduction to Machine Learning course by Andrew Ng on Coursera.

*Exercise 1:*
In this part of this exercise, I implemented linear regression with one variable to predict profits for a food truck in order to decide on a city to expand to next. The chain already has trucks in various cities and you have data for profits and populations from the cities.

- Plotted the data.
- **Implemented gradient descent.**
- **Implemented linear regression.**

<img width="612" alt="Screen Shot 2021-09-20 at 14 03 28" src="https://user-images.githubusercontent.com/30210990/134059587-8dde7555-e653-4e11-b61c-79a0ef4df47e.png">


*Exercise 2:*
Built a logistic regression model to predict whether a student gets admitted into a university. For each training example, I have the applicant's scores on two exams and the admissions decision. I have also predicted whether a microchip passes quality assurance test based on scores for 2 microchip tests. 

- **Plotted data.**
- **Implemented the sigmoid function.**
- **Implemented the cost function and gradient descent.**
- **Implemented regularized logistic regression.** For the microchip test model, I created mapped the features into polynomial terms of x1 and x2, up to the sixth power in order to create the circle shaped decision boundary.  
- **Evaluated logistic regression model.**
<img width="619" alt="Screen Shot 2021-09-21 at 12 26 29" src="https://user-images.githubusercontent.com/30210990/134218539-0e6a38a8-6fe4-461b-86cd-293aacf97c6a.png">
<img width="589" alt="Screen Shot 2021-09-21 at 12 28 15" src="https://user-images.githubusercontent.com/30210990/134218769-88e1f61f-5b93-405e-a1fa-160cf1961546.png">


*Exercise 3:*
For this exercise, I used logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. In the first part of the exercise, I applied logistic regression to one-vs-all classification.

- **Vectorized the cost function, gradient descent, and logistic regression for faster computing.**
- **Implemented one-vs-all multiclass logistic regression to recognize handwritten digits.** Implemented 10 logistic regression models for each digit and predicted the digit with the highest probability. 
- **Implemented feedforward propogation for a neural network to recognize handwritten digits.** Implemented feedforward propagation for a 3 layer neural network to predict handwritten digits with the weights provided.
<img width="397" alt="Screen Shot 2021-09-20 at 14 22 20" src="https://user-images.githubusercontent.com/30210990/134061863-fc3026f1-1883-4ec7-95c7-842d4a6754c2.png">


*Exercise 4:*
In this exercise, I implemented the backpropagation algorithm to learn the parameters for the neural network.

- **Implemented regularized cost function and gradient descent.**
- **Implemented backpropagation algorithm with random initialization and gradient checking to learn neural network parameters.**
- **Implemented regularized neural network to recognize handwritten digits.** (L2 regularization method)
<img width="579" alt="Screen Shot 2021-09-20 at 14 35 07" src="https://user-images.githubusercontent.com/30210990/134063356-6dc8ea15-45a0-4926-b965-73235d6a0c51.png">


*Exercise 5:*
In this exercise, I implemented regularized linear regression and use it to study models with different bias-variance properties. In the first half of the exercise, I implemented regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir. In the next half, I went through some diagnostics of debugging learning algorithms and examine the effects of bias vs. variance.

- **Implemented polynomial linear regression.**
- **Implemented cross validation to select lambda for regularization.**


*Exercise 6:*
In this exercise, I used support vector machines (SVMs) to build a spam classifier. In the first half of this exercise, I used support vector machines (SVMs) with various example 2D datasets. In the next half of the exercise, I used support vector machines to build a spam classifier.

- **Implemented SVMs with Gaussian kernels on datasets that are not linearly separable.** Gaussian kernel - similarity function that calculates the distance between 2 points. 
- **Trained SVMs to build a spam filter.**
<img width="561" alt="Screen Shot 2021-09-21 at 12 40 47" src="https://user-images.githubusercontent.com/30210990/134220526-9f006c1f-02d2-4734-a71e-84b2a6e69400.png">

*Exercise 7:*
In this exercise, I implemented the K-means clustering algorithm and applied it to compress an image by reducing the number of colors that occur in an image to only those that are most common in that image. In the second part, I used principal component analysis to find a low-dimensional representation of face images. 

In a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding. Our image contains thousands of colors, and in this part of the exercise, you will reduce the number of colors to 16 colors. By making this reduction, it is possible to represent (compress) the photo in an efficient way. Specifically, you only need to store the RGB values of the 16 selected colors, and for each pixel in the image you now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities). 

- **Implemented K-means clustering to compress an image.** Reduced the number of bits per pixel location from 24 to 4, by storing the representation of 16 main colors in a dictionary and representating the 16 colors using 4 bits per pixel location when compressing the image. Achieved a compression by a factor of 6 - from 128\*128\*24=393,216 bits to 24\*16 +128\*128\*4=65,920. 
<img width="618" alt="Screen Shot 2021-09-20 at 11 31 55" src="https://user-images.githubusercontent.com/30210990/134039392-53c38dd1-7e57-4e40-88c6-347d654af45f.png">

- **Implemented PCA to find a low-dimensional representation of face images.** Used PCA on black and white images (grayscale) of faces and reduced dimensionality of image representation from 1,024 to 100 - some fine details were lost, but the main face characteristics were preserved. 
<img width="739" alt="Screen Shot 2021-09-20 at 11 31 31" src="https://user-images.githubusercontent.com/30210990/134039320-c116b817-1a83-41ac-a648-040c4b00e069.png">


*Exercise 8:*
In this exercise, I implemented the anomaly detection algorithm and applied it to detect failing servers on a network. In the second part, I used collaborative filtering to build a recommender system for movies.

- **Implemented anomaly detection.** Fitted a Gaussian distribution on a dataset with features on server throughput and latency - found mean and variance for each feature. By the means of cross validation on a validation data set and F1 scores, I found the optimal value for maximum probability to be used as a threshold for detecting anomalies. 

<img width="565" alt="Screen Shot 2021-09-20 at 11 58 47" src="https://user-images.githubusercontent.com/30210990/134043084-9617006b-ad8c-4206-b2d1-e29c1772de12.png">

- **Implemented a recommender system for movies.** Implemented a collaborative filtering learning algorithm to predict ratings for movies by specific users. The algorithm employs gradient descent on matrix X (movies for rows and features for columns) and Theta (weights learned for each user); with the aid of the matrix Y (ratings of users as columns for movies as rows) and matrix R (binarty matrix with 1 for seen movies and 0 for not seen, movies as rows and users as columns). 

<img width="735" alt="Screen Shot 2021-09-20 at 13 51 04" src="https://user-images.githubusercontent.com/30210990/134057967-2b6e0c54-2d04-457f-977f-8ff6b4706f9c.png">
