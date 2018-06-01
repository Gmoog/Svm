## Python implementation of Linear SVM with Squared Hinge Loss

We look at how to implement the Linear Support Vector Machine with a squared hinge loss in python.

The code uses the fast gradient descent algorithm, and we find the optimal value for the regularization parameter using cross validation.

### Files

The code is broadly divided into the following submodules:

 * *Svm.py* implements all the code for calculating the objective, gradient, and the fast gradient algorithm.
 * *simulated_data.py* generates simulated data and runs the algorithm implemented in Svm for this data.
 * *real_world_data.py* runs the algorithm on the Spam data set downloaded from the book Elements of Statistical Learning.
 * *compare.py* compares the output for the Spam dataset from Scikit learn and the code in Svm.py


### Data

We use the Spam dataset from the book *The Elements of Statistical Learning*, as the real-world-dataset. 
The dataset can be accessed from (https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data).
The simulated data set is generated using np.random.normal and np.random.uniform functions and consists of 100 observations and 10 features.

### Installation

To run the code for the existing datasets, make sure the files are in the same folder and run the commands:

 * *python real_world_data.py* to run the algorithm on the Spam dataset.
 * *python simulated_data.py* to run the algorithm on the simulated dataset.
 * *python compare.py* to compare the output from scikit learn and our code for the spam dataset.




