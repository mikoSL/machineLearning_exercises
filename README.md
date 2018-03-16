# Learning note from the book "Hands-on Machine Learning with Scikit Learn and Tensorflow" by Aurelien Geron. 
---
## Chapter 2 Essentials: workflow of machine learning project
### 1. Glance data
* data structure
* data information
* data description

### 2. Split train/test set 
## (YES! already in this phase!)

### 3. Explore data
* visualization
* correlation
* possible attribute combination

### 4. Prepare for machine learning
* data cleaning (missing data etc.)
* categorical data/text
* transformation
* feature scaling
## (Build an automated transformation pipeline!)

### 5. Build Macine Learning models (3-5 models)
## (Compare their results on TRAINNING set!)

### 6. Fine tune for best model
* grid search
* randomized search (for large hyperparameters)
* ensemble model

### 7. Analyze the best model and its errors
* e.g. check the importance of each attribute

### 8. Evaluate on test set
## (NOT tweak hyperparameter to make the number look good on test set! The improvement is unlikely to generalize to new data.)

### 9. Launch project
* plug production input dat to system
* write test

### 10. Monitor and maintain the system
* automated monitor code
* evaluate input data quality
* human evaluation pipeline
* automated regular model train with new data

## Chapter 3 Essentials: classification
### 1. Classifier
* Binary Classifier (SVM support vector machine, or linear classifier)
* Multiclass Classifier (random forest classifier, naive bayes classifier)
## (which strategy? OvA- one versus all or OvO - one versus one)
* Multilabel Classifier
## (When labels are not equally important, choose "weight" method in f1_score.)
* Multioutput Classifier

### 2. Performance Evaluation
* cross validation
## (NOT suitable for skewed data!)
* confusion matrix 
## (precision/recall tradeoff!)
* ROC-AUC curve (Receiver Operating Characteristic) (Area Under the Curve)
## (PR(precision Recall curve) or ROC curve, which one? Choose PR when positive case is rare and you care more about the false positives than the false negatives)

## Chapter 4 Essentials: regression, optimal algorithem and regularization)
### 1. Gradient Descent (GD)
* 1. random initialization
* 2. iterate to decrease cost function (different learning rate)
* 3. converage to minima (local minima or global minima)
## (GD used for linear regression is convex function--no local minima, one global minimal and continuous function--slope never chane abruptly)

#### 1.1. Batch Gradient Descent (BGD)
* 1. run on full train set!
* 2. Q: How many iteration should be used? A: Set very large number of iteration, interrupt the algorithem when gradient vector becomes tiny--tolerance)
* 3. convergence rate. when cost function is convex and slope rate not change abruptly, convergence rate of BGD with a fixed learn rate is O(1/iteration), meaning that if divide tolerance by 10, iterate will be 10 times (10*iteration)

#### 1.2. Stochastic Gradient Descent (SGD)
* 1. run on a random pick-up instance.
* 2. better to find global minima.
* 3. never settle at minimum because instance is random selected. 
## (solution: simulated annealing--gradually reduce learning rate)
* 4. learning schedule: function to determine what is the learning rate.
* 5. epoch (one round of m iteration)

#### 1.3. Mini-batch Gradient Descent (MBGD)
* 1. run on small random sets of instance.
* 2. + performance boost from hardware optimization of matrix operation, especailly when using GPUs.
* 3. + closer to minimum than SGD.
* 4. - harder to escape from local minima.

### 2. Polynomial regression
* 1. in sklearn, PolynomialFeature (degree = d) transform array contain n feature to (n+d)!/d!n!
* 2. capable of finding relationship between features.
* 3. learning curves. underfit--not hlep to feed more training data. overfit--feed it more train data until the validation error reaches the training error.
## (bias/variance tradeoff! Bias--Underfit! Variance--Overfit!)
* 4. bias: wrong assumption, assume it is linear but it is quadratic. A high bias model is most likely to UNDERFIT train data.
* 5. variance: model's excessive sensitivity to small variaion in training data. A model with many degrees of freedom is likely to have high variance. It normally causes OVERFIT.
* 6. irreducible error (caused by noise in data)

### 3. Regularize linear model
* 1. One way to regularize polynomial regress is to reduce degree.
## (For linear model, regularization id done by constrai weights of model!)
* 2. ridge regression (L2 norm of the weight vector)
## (scale data when apply ridge regression! add it during train, but evaluate the model performance using unregularized performance measure!)
* 3. lasso regression (tent to completely eliminate weight of the least important feature)
## (automatically perform feature selection and outputs a sparse model! Behave erratically when number of feature is greater than number of train instance or when several feature are strongly correlated!)
* 4. Elastic Net(perfered over lasso)

### 4. Early stop
* One way to regularize iteration learning algorithem (eg. GD) is to stop train as soon as the validation error reaches a minimum.

### 5. Logistic regression (caculate probability that one instant belong to a particular class)
* 1. sigmoid function.
* 2. log loss
## (cost function--log loss is convex , meaning it has one global minima)

### 6. Softmax regression (multinomial logistic regression)
* 1. multiclass not multioutput, meaning it can not recognize multiple people in one picture, only can mutually exclusive classes such as different plant)
* 2. normalized exponential (prob softmax function--compute score for each class k)
* 3. argmax
* 4. cross entropy (min cost function): measure how well a set of estimated class prob match the target classes.
## (cost function--log loss is convex , meaning it has one global minima)
