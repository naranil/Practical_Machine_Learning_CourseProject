Practical Machine Learning - Course Project
========================================================
## Introduction
Nowadays some devices can record different kind of body motion with a high accuracy. Those records can help researchers and sport professionals to understand better which features matter for a movement execution.

In this course project, we'll try to use machine learning tools described in Practical Machine Learning lecture. The data used was kindly given by [Groupware](http://groupware.les.inf.puc-rio.br/har). It's a collection of records of accelerometers posed on different part of a participant arm while he's performing barbell.

This study has been done on Mac OS X, version 10.9.3, with a 2.3 GHz Intel Core i5 processor and 4Go of memory. I used the statistical language R and its famous machine learning package, caret.

```r
library(caret)
```

## Data cleaning and processing
First we download the two data sets given on the course page: 

```r
## Training Set
if (!file.exists("./Data/trainingset.csv"))
{
  download.file(
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    destfile="./Data/trainingset.csv", 
    method="curl")
}

## Test Set
if (!file.exists("./Data/testset.csv"))
{
  download.file(
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
    destfile="./Data/testset.csv", 
    method="curl")
}
```
In this study we'll consider mostly the training set, since I'll evaluate the efficiency by cross validation. However, I'll mention the results of the prediction on the test set at the end of this study. 
Then, we use **read.csv** to create two data sets corresponding to the the train and test sets in cs format:

```r
train <- read.csv("./Data/trainingset.csv")
test <- read.csv("./Data/testset.csv")
## Make sure the two data sets columns have the same names
names(test) <- names(train) 
```
If we have a look to the train set, we can notice that a lot of columns are empty or filled with NA values. The following code shows that 100 columns have less than 10% of values.

```r
nb_empty <- 0
for (name in names(train)){
  if (sum(!is.na(train)[,name])/length(train[,name]) < 0.9 ||
        sum(!train[,name]=="")/length(train[,name]) < 0.9){
    
    nb_empty <- nb_empty + 1
    
  }
}

## Number of columns with less than 90% of values:
nb_empty
```

```
## [1] 100
```
After several simulation *a posteriori*, I decided to keep only the columns which have 100% of values.

```r
trainclean <- train
for (name in names(train)){
  if (sum(!is.na(trainclean)[,name])/length(trainclean[,name]) != 1 ||
        sum(!trainclean[,name]=="")/length(trainclean[,name]) != 1){
    
    trainclean[,name] <- NULL
    
  }
}
```
It can also be noticed, that 7 other columns don't give any information in the analysis: **X** is just an increment variable, **user_name** corresponds to the participant's first name, the **raw_timestamp_part_1**, **raw_timestamp_part_2**, **cvtd_timestamp**, **new_window** and **num_window** are related to time variables which shouldn't have a link with the experiment results.

```r
trainclean <- trainclean[,8:60]
```
We ends up with a cleaned data set with 19622 observations and **53 variables** (52 features and 1 outcome variable). We can now go to the machine learning part of our study. 

## Feature selection
Even if we cleaned the data set, a 53var. x 19622obs. set is still pretty big. Some features contain more information than others. It's possible to evaluate the features that contain a certain ratio of variance thank to the correlation matrix.

```r
corrMatrix <- abs(cor(trainclean[,-53]))
diag(corrMatrix) <- 0
features <- c(names(which(corrMatrix > 0.8, arr.ind=T)[,1]), "classe")
## Number of features with a correlation of more than 80%
length(features)
```

```
## [1] 39
```
The threshold of 0.8 helps to keep the main part of the variance.

We then create the two train and test data sets of **39 variables** that will be used by the machine learning algorithms.

```r
training <- trainclean[, features]
testing <- test[, features]
```

## Number of observations retained
The algorithms used can take quite some time in terms of computing. 19622 observations is a huge quantity of data, we should then consider to remove a certain part of those observations. Here is a plot of the learning curve with the Random Forest algorithm in order to evaluate how much data we'll keep.
![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9.png) 

## Algorithm used and my model
I'll use the Random Forest algorithm which is well known for giving the best accuracies.

We randomly select 10000 of observations. 

```r
randomsample <- sample(nrow(training), 10000)
training <- training[randomsample,]
```

The model is evaluated with the **train** function of the caret package. Since the computation takes some time, the following line of code won't be evaluated. I kept the **modFit** variable in a .RData file that will be loaded in a hidden line of code readable in the source file.

```r
modFit <- train(training$classe~.,data=training, method="rf")
```




```r
modFit$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9615 0.9513   0.003827 0.004825
## 2   20   0.9547 0.9427   0.003942 0.004976
## 3   38   0.9441 0.9293   0.007888 0.009973
```

## Estimation of the error with cross-validation
To estimate the accuracy and thus the error rate, we can make a 5-folds cross validation

```r
control <- trainControl(method = "cv", number  = 5)
modFitCV <- train(training$classe~.,data=training, method="rf", trControl=control)
```




```r
modFitCV$resample
```

```
##   Accuracy  Kappa Resample
## 1    0.971 0.9633    Fold1
## 2    0.962 0.9519    Fold2
## 3    0.965 0.9557    Fold5
## 4    0.963 0.9532    Fold4
## 5    0.968 0.9595    Fold3
```

```r
modFitCV$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9658 0.9567   0.003705 0.004687
## 2   20   0.9616 0.9514   0.006827 0.008648
## 3   38   0.9533 0.9409   0.006086 0.007725
```

The we obtain an accuracy of 96.6% which seems pretty good for a 5 classes classification problem. Thus, we can use this model to make some predictions on our test set.
## Preprocessing and PCA
In this study, the data was not normalized and Principal Component Analysis was not used. Indeed, if we keep 3000 observations and use PCA with the Random Forest algorithm, it seems that we get worse results:

```r
modFit3000 = train(training$classe~.,data=training, methode='rf')
modFit3000PCA = train(training$classe~.,data=training, methode='rf', preProcess="pca")
```




```r
modFit3000$results
```

```
##   mtry Accuracy  Kappa AccuracySD KappaSD
## 1    2   0.8048 0.7521    0.02496 0.03150
## 2   20   0.7975 0.7435    0.02754 0.03467
## 3   38   0.7830 0.7252    0.03014 0.03789
```

```r
modFit3000PCA$results
```

```
##   mtry Accuracy  Kappa AccuracySD KappaSD
## 1    2   0.5852 0.4733    0.03306 0.04205
## 2   20   0.5582 0.4396    0.03542 0.04500
## 3   38   0.5575 0.4388    0.03337 0.04247
```
## Test set
To conclude this analysis, I briefly present the results I got from the test set. To predict the outcome for our testing set, we simply use the command:

```r
predict(modFit, newdata=testing)
```
After submission to Coursera's server, I got one false prediction over 20, which represents an accuracy of **95%**. This accuracy was expected thank to our error analysis done before.

----
