---
description: some traditional machine learning algorithms
---

# Traditionals

## Survey Papers / Repos

* Top 10 algorithms in data mining. [\[ICDM'06\]](http://39.104.72.142:802/algorithms/10Algorithms-08.pdf)
* [josephmisiti/awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning)

## Resources

* [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) by Andrew Ng

## Tasks

### Supervised

* Linear Regression

$$
y=ax+b\\
L(y,\hat{y}) = (y-\hat{y})^2
$$

* Logistic Regression

$$
y=\frac{1}{1+e^{-(ax+b)}} \\
L(y,\hat{y}) = -\hat{y}\log y - (1 - \hat{y}) \log (1-y)
$$

* Support Vector Machine \(SVM\)
  * Process: Lagrange -&gt; Dual Problem -&gt; SMO

$$
\min \frac{1}{2} ||w||^2  \\
\text{s.t.}~y^{(i)}(w^{T}x^{(i)}+b) \geq 1, i=1,...,m
$$

* K Nearest Neighbor \(kNN\)
* Decision Tree
* Random Forest
* Naive Bayes
* Expectation-Maximization \(EM\)
* Linear Discrimant Analysis \(LDA\)
* Gradient Boosting Tree \(GBDT\)

### Semi-supervised

### Weakly-supervised

### Unsupervised

* Clustering
  * K-means
  * Mean-shift
  * DBSCAN
* Principal Component Analysis \(PCA\)
* Latent Dirichlet allocation \(LDA\) Topic Modeling

## Others

### Ensemble

* K-Fold Cross Validation
* Bagging
* Boosting

### Metrics

|  | True Samples | False Samples |
| :--- | :--- | :--- |
| Predict True | True Positive | False Positive \[Type I Error\] |
| Predict False | False Negative \[Type II Error\] | True Negative |

* Precision and Recall
  * $$\text{Precision} = \frac{\text{TP}}{\text{TP} +\text{FP}}$$
  * $$\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}$$
* F1 Score
  * $$\text{F1 score} = 2 \cdot\frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} +\text{Recall}}$$
* Receiver Operating Characteristic \(ROC\)
  * $$\text{TPR} = \frac{\text{TP}}{\text{TP}+\text{FN}}$$
  * $$\text{FPR} = \frac{\text{FP}}{\text{FP}+\text{TN}}$$
* Area Under ROC \(AUC\)
* Confusion Matrix

## Reference

* SVM: [https://www.cnblogs.com/jerrylead/archive/2011/03/13/1982639.html](https://www.cnblogs.com/jerrylead/archive/2011/03/13/1982639.html)



