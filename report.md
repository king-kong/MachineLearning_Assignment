Machine Learning Assignment
========================================================

**Summary**

This document summarizes the analysis that was perform to build a predictive model for data from a fitness gadget to predict exercise class.   

The final classification method used was Random Forests with cross validation of 10-folds.   The data set provided was too large for my laptop (and i started this assignment too late!).   Hence, I ran the training only a randomly sample of the partitioned training set (10000 records).   As a result, the model had a poorer than expected performance compared to other models.  After creating the model, it was used to predict on the testing set and a confusion matrix was created (Accuracy :0.9628, 95% CI : (0.9576, 0.9675)).   The number of variabled tried at each step was 2. 

**Data Analysis**

Importing the data:


```r
ass <- read.csv("pml-training.csv")
```


Upon inspecting the data, it was found that there were many columns there were N/A or are irrelevant to exercise class (e.g. subject name).   Only the measurements from the gadget were used.  'classe' was already a factor after the import.


```r
ass_clean <- ass[, c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", 
    "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y", 
    "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_forearm", 
    "pitch_forearm", "yaw_forearm", "total_accel_forearm", "gyros_forearm_x", 
    "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", 
    "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", 
    "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", 
    "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", 
    "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
    "total_accel_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", 
    "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", 
    "magnet_dumbbell_y", "magnet_dumbbell_z", "classe")]
```


On this set of data, partition it into training and testing.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(e10701)
```

```
## Error: there is no package called 'e10701'
```

```r
inTrain <- createDataPartition(y = ass_clean$classe, p = 0.7, list = FALSE)
training <- ass_clean[inTrain, ]
testing <- ass_clean[-inTrain, ]
```


Because of time contraints, a 10000 random of the training data set was taken for the model building. 


```r
set.seed(12345)
training_sample <- training[sample(1:nrow(training), 10000, replace = FALSE), 
    ]
```


Create a model based on the training_sample set.  PCA was use for preprocess to remove unnecessary features and cross validation was used to better calculate the accuracy.   The default 10-fold was used.   



```r
library(caret)
library(e10701)
```

```
## Error: there is no package called 'e10701'
```

```r
rd_fit <- train(classe ~ ., training_sample, method = "rf", preProcess = c("pca"), 
    trControl = trainControl(method = "cv"))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```


**Model Out-of-Sample Error and Model Diagnostics Discussion**

Since there are plenty of samples available, it is possible to further increase K until the Accuracy decreases due to increase in bias.  It will decrease variance, hence a smaller .95 confidence interval to ensure the Accuracy can be counted on for statistical inference.

In terms of gauging Accuracy, the following code snippet was used to create a confusion matrix on the testing set.


```r
library(caret)
library(e10701)
```

```
## Error: there is no package called 'e10701'
```

```r
testing_results <- predict(rd_fit, newdata = testing)
confusionMatrix(testing_results, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1657   26    8    3    1
##          B    7 1084   23    0    9
##          C    7   22  979   62   19
##          D    3    4   13  896    5
##          E    0    3    3    3 1048
## 
## Overall Statistics
##                                         
##                Accuracy : 0.962         
##                  95% CI : (0.957, 0.967)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.952         
##  Mcnemar's Test P-Value : 9.03e-10      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.990    0.952    0.954    0.929    0.969
## Specificity             0.991    0.992    0.977    0.995    0.998
## Pos Pred Value          0.978    0.965    0.899    0.973    0.991
## Neg Pred Value          0.996    0.988    0.990    0.986    0.993
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.184    0.166    0.152    0.178
## Detection Prevalence    0.288    0.191    0.185    0.156    0.180
## Balanced Accuracy       0.990    0.972    0.966    0.962    0.983
```

