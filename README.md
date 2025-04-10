# My MSc Dissertation Research
A Comparative Study of ‘Cost Sensitive Learning’ Techniques On Machine Learning Models To Decrease Misclassification Costs In Loan Default Prediction.

This is a comparative based study to examine performance of performance of using cost sensitive learning (CSL) techniques in predicting loan defaults. Two approached in cost sensitive learning: instance weighing resampling (CSL – R) and thresholding in algorithm model (CSL – A). These techniques are applied to three machine learning models - Logistic Regression, Decision Trees and Extreme Gradient Boosting (XG-Boost). CSL – R has a superior performance in logistic regression and decision tree models and XG-Boost works well with CSL – A approach. Overall, XG-Boost CSL – A approach was the found as an ideal CSL technique to predict loan defaults. It shows lowest misclassification errors and highest AUC value among the models.

[Keywords]: Loan Default Prediction, Machine Learning, Cost Sensitive Learning, CSL Instance weighing, CSL Thresholding, R 


This study is conducted to compare two cost-sensitive approaches using three machine learning models. The framework proposed has four phases:
1. Literature Review : Identifying research gap & related literature on the topic
2. Methodology : Creating a structured research methodology for in depth analysis
3.	Data preparation & modelling : Preparing dataset for ML modelling
4.	Results : Understanding results of each ML model  
6.	Analysis : Conclusion of the research
7.	Research Limitations
8.	Future research 

The detailed research flow is shown as follows:
![image](https://github.com/user-attachments/assets/6b116c63-a8b9-4572-a92c-227f47d9eb5b)
 
 
Under data pre-processing, the dataset is prepared for ML modelling. It includes selection of variables (also known as feature selection), data preparation including handling of missing values & dummy variable creation. The processed dataset is then split into two parts – training dataset for model learning purpose and testing dataset for model validation purpose. Thereafter, the training dataset is used to train models for both CSL methods:

#### Method 1: Cost Sensitive Learning with Resampling (CSL-R): 
In this method, costs are minimized by giving higher importance to less costly class during the model training phase.  This is accomplished by adding class weights to rebalance the cost associated with classes. ‘Weighted cost-sensitive wrapper’ is introduced during the model training. 

#### Method 2: Cost Sensitive learning at Algorithm level (CSL-A): 
Misclassification costs are introduced within the model algorithm after model training. This is carried out by adjusting the threshold values of the model’s decision boundary, without affecting the learning process.

In comparing both the methods, this study aims to find effective method to minimize misclassification costs in imbalanced loan datasets. This will help in decreasing predictive bias in imbalanced datasets. This research intends to achieve the following objectives:
1.	To compare performance of CSL techniques (CSL – R & CSL – A) in understanding its effectiveness in reducing misclassification costs.
2.	To find the best CSL model in predicting default & decreasing risk exposure.

To achieve the first objective, three ML models, namely logistic regression, decision tree & extreme gradient boosting, are chosen based on the literature available on cost sensitive models and its effectiveness in handling misclassification errors. These three models will be trained on both techniques. Their performance will be evaluated based on calculating total costs and average misclassification cost. The least cost model with overall higher performance will be the best model. 
The second objective will be to find the best performing model among the six ML models trained, i.e. CSL – R Logistic regression, CSL – R Decision tree, CSL – R XG-Boost, CSL – A logistic regression, CSL – A decision tree, CSL – A XG-Boost.  
Its real-life applicability such as the model’s capability to minimize the credit risk exposure of an organisation will determine its effectiveness. In this case, the focus will be to minimize probability of default. 

// ** More in detail covered in Methodology document attached ** //

## Dataset for analysis
In this study, an online USA peer-to-peer (P2P) lending platform dataset is utilised to predict loan defaults. It is available publicly on their website.
Source: https://www.lendingclub.com/info/download-data.action.
The dataset consists of loans issued between 2007 and 2011. These loans are unsecured personal loans that cost from 1,000 to 40,000 USD. The average loan period is 3 years. The total number of loans are 42,535 with 116 attributes.
The attribute of interest is ‘loan default’. It has a binary values ‘Y’ or ‘N’. ‘Y’ if the loan is defaulted. ‘N’ if the loan is not defaulted. There is an overall 18% default rate in the data. The default distribution annually is shown in figure below:

![image](https://github.com/user-attachments/assets/fe6021b1-56b7-4935-a1b5-6f8fb4bccb2f)

## Software Packages used in the research
R programming language (R core team, 2023) was utilized to perform data manipulation and predictive machine learning (ML) modelling in this study. The analysis was carried out in the R Studio (Posit team, 2023). The packages used were:
1. ‘caret’ (Kuhn M, 2008)	: confusion matrix and cost matrix.
2. ‘dplyr’; ‘tidyr’ (Wickham et al., 2023):	Data manipulation.
3. ‘ggplot2’ (Wickham H, 2016):	Data visualization.
4. ‘rpart’ (Therneau and Atkinson, 2022):	Random forest model building.
5. ‘xgboost’ (Chen et al.,2024):	XG-Boost model building.
6. ‘proc’ (Binder et al.,2019):	Plotting ROC and AUC values.
7. ‘mlr3’ (Kuhn M, 2008) & ‘mlr’ (Bischl et al., 2016): Create learning tasks, modelling and prediction.
9. ‘stats’ (R core Team, 2023):	Logistic regression model building.
10. ‘mlr3learners’ (Lang et al.,2024):	ML task learners for modelling.
11. ‘mlr3measures’ (Becker M & Lang M, 2024):	Built in performance measures for model evaluation.
12. ‘mlr3tuning’ (Becker, Lang M, Richter et al., 2024): Optimise hyper parameters of the models integrating with the mlr3 framework.
13. ‘mlr3verse’ (Becker M, Lang M, Schratz P, 2024): Resampling strategies for model evaluation.
14. ‘mlr3misc’ (Lang & Schwartz, 2024): Data splitting.

## Model Output
The three classification models logistic regression, decision tree and XG-Boost, were optimised using tuned parameters, tuned weights and empirical threshold values. The optimised models were then trained to develop relationships using the data in the training dataset. Thereafter, the testing dataset is utilised to examine the performance of the trained model in predicting defaults. 

To understand overall model’s performance in differentiating positive and negative classes (i.e. defaults and non-defaults), ROC plot for both the best models from each technique is shown below:

![image](https://github.com/user-attachments/assets/2647f461-a3c1-4620-8fe8-0ee873e7ae24)

## Model Analysis
Credit risk modelling is a mix of both understanding technical machine learning aspects as well as business side of lending. Machine learning models may seem easy to build, but in real life, its application and achieving its business objective is a difficult task. In this study, two CSL techniques were compared to find which models are able to decrease the misclassification costs and increase model’s accuracy in predicting defaults. CSL- R technique uses instance weighing to incorporate misclassification costs at training level, while CSL – A technique uses thresholding to optimise decision parameters as misclassification costs are introduced at algorithm level after the model training is completed. 
Both the techniques have their advantages. CSL – R technique performs better with logistics regression and decision tree models. Since logistic regression optimizes a probabilistic function, modifying instance weights during the training directly influences loss minimization. By assigning higher weights on misclassification during the model training, it equips the logistic regression model to adjust the decision boundary to give importance to minority class (defaults). This in turn increases the recall metric. Similarly, instance weighing impacts the decisions of node splits, pushing the tree to focus on high cost classes (i.e. the minority class). Hence it provides the tree not to over fit during the training process, improving model generalisation.    
On the other hand, XG-Boost model performs well with the CSL – A technique. XG-Boost model’s boosting algorithm with threshold tuning intelligently shifts the decision boundary according to the cost matrix, increasing its importance to minority class. Hence, the model improves its minority class pattern recognition.  

Lending organisations have a simple business objective to minimize its risk exposure and maximize its revenue. To minimize the risk exposure, organisations monitor loan application acceptance rates and rejection rates. These rates are derived from the prediction model. In this case, CSL – A has false positive (FP) = 335. Therefore, if this model is used in default prediction then  335/9134%, i.e. 3.6% of loan application cases will be rejected. Based on organisation’s risk appetite, either models logistic regression or XG-Boost will be effective in decreasing misclassification costs in an imbalanced dataset. 

## Limitations in this research
One of the key dependencies in the cost sensitive learning is finding an ideal cost matrix. There are two approaches in finding the ideal misclassification costs. First is with the help of business objectives in estimating profits and loss for each disbursed loan. Second method is industry expert based approach, where an expert determines the misclassification cost. Both the methods are complicated in real life. Not having an ideal cost matrix limits the use of such techniques in large scale projects.
The data used in the study has certain limitations. Dataset may contain hidden errors or human errors during the creation of the data. Although, the dataset is relevant and shows good performance in predicting defaults, the latest data may have been relevant in the curremt scenario. Latest data was not available in the public domain for research purposes. Additionally, due to budget constraints, repositories such as European Datawarehouse (EDA) could not be partnered with to obtain latest datasets. 
Lastly, due to computational constraints, only 3-fold cross validation resampling strategy was able to perform to safeguard processing overload or R Studio application crash. A thorough 5-fold cross validation, hold out methods, etc for higher computational validation could not be performed.

## Future research
The study is limits itself to binary ML classification applications. Future research in this area may explore with a multi-class problem in predicting defaults using different loan grades. Loan grades from A to G can be sub-divided to predict defaults within in each grade. Hence, cost matrix will be unique for every loan grade. Multi-class model comparisons will create an overall robustness test in using cost sensitive learning. 
Moreover, future research could explore multiple datasets and applications in different field of study such as fraud detection, healthcare, etc. This will widen the scope and effectiveness of cost sensitive learning techniques. 
A possible extension of the current study to integrate a more robust model validation techniques such as back testing, scenario analysis based on the changing consumer behaviour and stress-testing based on changing macroeconomic conditions such as increasing cost of living. This research area can further be extended in performing sensitivity analysis with cost-sensitive classifiers. For example, changing the credit policies or approval rate to test the model prediction may be a robust model validation of CSL models in handling extreme changes in data. 






