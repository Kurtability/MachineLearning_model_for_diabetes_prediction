# MachineLearning_model_for_diabetes_prediction

## Introduction:
Many people nowadays suffer from diabetes, a group of metabolic diseases where there are high blood sugar levels over a long period of time. Patients suffering from diabetes exhibit symptoms such as frequent urination, increased thirst, and increased hunger. The objective of this study is to train an accurate model based on the diagnostic measurements within the dataset which aids the doctors to diagnose whether or not a patient has diabetes.
The study is important because once the accuracy of the predictions of the trained model becomes higher than the accuracy of the diagnoses of the doctors, the model can then be used as a robust tool by the doctor to diagnose patients more efficiently and more accurately. Therefore, more lives will be saved.

## Data:
The title of the dataset is called Pima Indians Diabetes Database and the original owner of the dataset is National Institute of Diabetes and Digestive and Kidney diseases. However, the dataset has been modified which replaced all missing values with averages. Furthermore, all patients within the dataset are female at least 21 years old of Pima Indian heritage. The dataset consisted of 8 medical predictor variables and 1 class variable which is the outcome and the number of instances for the dataset is 768. The variables involved in the dataset are number of times the patient has been pregnant, plasma glucose concentration a 2 hours in an oral glucose tolerance test, diastolic blood pressure(mm Hg), triceps skin fold thickness(mm), 2-hour seem insulin(mu U/ml), body mass index, diabetes pedigree function and age(years).
CFS stands correlation feature selection, its essential in the field of machine learning because it provides a way to create a model that only includes the most informative features. In the world of data science, there are countless amounts of features sometimes, CFS can resolve this dilemma. There are many benefits of CFS, including the reduction of the variation of the model, and the reduction of the computational cost and the time of the training model. Furthermore, CFS increases the reliability, stability, and classification accuracy of the model.

## Results and discussion
(No feature selection is NFS, correlation feature selection is CFS)
ZeroR: NFS: 0.424 CFS: 0.424
1R: NFS: 0.698 CFS: 0.698
1NN: NFS: 0.679 CFS: 0.69
5NN: NFS: 0.741 CFS: 0.741
NB: NFS: 0.747 CFS: 0.757
DT: NFS: 0.725 CFS: 0.736
MLP: NFS: 0.755 CFS: 0.757 
SVM: NFS: 0.752 CFS: 0.762
RF: NFS: 0.742 CFS: 0.747
My1NN: NFS: 0.675 CFS: 0.695
My5NN: NFS: 0.746 CFS: 0.750
MyNB: NFS: 0.745 CFS: 0.768

From looking at the results, we can summarise that the performance of the classifier with feature selection is usually higher than the no-feature- selection. However, for ZeroR, 1R, and 5NN, the accuracy outputs are the same between the CFS data and no-feature-selection. From the results of Weka, we can see that the CFS SVM algorithm has the highest accuracy among the rest. However, if we were to compare highest performer of Weka to our implementations of K-NN and NB, it shows in the results that MyNB CFS has the best performance comparing to Weka’s classifiers.

By using the correlation based feature selection in Weka, we managed to generate another csv file with the corresponding CFS attributes. Although, the selective subset does not make intuitive sense to us by just reading the data descriptions of the subset. However, the results demonstrates that having CFS in the algorithm usually improves the accuracy of the model. Furthermore, CFS is proved by the results to reduce the computational cost and time of a training model.

## Conclusion and future work:
The Naive Bayes’ and K-Nearest Neighbor algorithms we have provided, have a precision which is comparable to the one produced using the same classifiers in WEKA.

Of the tested classification methods, the NB code that we have provided shows the highest precision and would be a good model to further work upon. Some other viable models are SVM, KNN and DT.
The NB algorithm has issues classifying data with an attribute value combination it has not been trained upon. To build a more robust model we’d like to work on training a neural network to help us predict if a patient has diabetes.
