# ROC plotting
############################### combined ROC ##############
#Logreg
roc_logreg <- roc(as.numeric(testData$default), as.numeric(predict_log_th2$data$response), curve=TRUE)
round(auc(roc_logreg),3)
#xgboost
roc_xgboost <- roc(as.numeric(testData$default), as.numeric(predict_xgbmodel$data$response), curve=TRUE)
round(auc(roc_xgboost),3)

ggroc(list("Logistic Regression Model CSL- R" = roc_logreg, "Extreme Gradient Boosting Model CSL- A" = roc_xgboost)) +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  labs( x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR") +
  theme(legend.position = "bottom")