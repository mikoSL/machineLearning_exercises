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
