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
## (NOT tweak hyperparameter to make the number look good on test set! The improvement is unlikely to generalize to new new data.)

### 9. Launch project
* plug production input dat to system
* write test

### 10. Monitor and maintain the system
* automated monitor code
* evaluate input data quality
* human evaluation pipeline
* automated regular model train with new data
