## Python implementation of Linear Support Vector Machine with Squared Hinge Loss

We look at how to implement the Linear Support Vector Machine with a squared hinge loss in python.

The code uses the fast gradient descent algorithm, and we find the optimal value for the regularization parameter using cross validation.

### Files

The code is divided into the following 5 files:

 - svm.py implements all the code for calculating the objective, gradient, and the fast gradient algorithm
 - simulated_data.py is a demo file, which launchs the method on a simple simulated dataset.
 - real_world_data.py is a demo file, which launchs the method on a real-world dataset.
 - sklearn_vs_ourcode.py compares the results from this implementation and scikit-learn.
 - demo_compare.py is demo of the comparison on a simulated and a real-world dataset.



### Datasets

We use the Spam dataset from the book *The Elements of Statistical Learning*, as the real-world-dataset. You can access the dataset from [the website of the book](https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets).



