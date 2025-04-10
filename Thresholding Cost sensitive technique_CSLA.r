# Thresholding Cost sensitive technique
################################# CSL A ########################################

library("mlr")
library("BBmisc")
library("ParamHelpers")

# class weights
class_weights <- table(trainData$default2)
weights <- ifelse(trainData$default2 == "1", class_weights[2] / sum(class_weights), class_weights[1] / sum(class_weights))
summary(as.factor(weights))

trainData$default2 <- NULL
testData$default2 <- NULL

credit.task = makeClassifTask(data = trainData, target = "default", weights = weights)
credit.task = removeConstantFeatures(credit.task)
credit.task


###################### Logistic reg ########################
learner = makeLearner("classif.logreg",
                      predict.type = "prob")
mod = train(learner, credit.task)
predict_log = predict(mod, newdata = testData)
predict_log

predict_log_th = setThreshold(predict_log, th)
predict_log_th

parallelStop()
parallelStartSocket(4)
rin = makeResampleInstance("CV", iters = 3, task = credit.task)
lrn = makeLearner("classif.logreg", predict.type = "prob", predict.threshold = th)
r = resample(lrn, credit.task, resampling = rin, measures = list(credit.costs, mmce), show.info = FALSE)
r


d = generateThreshVsPerfData(r, measures = list(credit.costs, mmce))
plotThreshVsPerf(d, mark.th = th)

tune.res = tuneThreshold(pred = r$pred, measure = credit.costs)
tune.res


tune.res2 = tuneThreshold(pred = r$pred, measure = mmce)
tune.res2



th2 = 0.6232963
predict_log_th2 = setThreshold(predict_log, th2)
predict_log_th2

performance(predict_log, measures = list(credit.costs, mmce))
performance(predict_log_th, measures = list(credit.costs, mmce))
performance(predict_log_th2, measures = list(credit.costs, mmce))

confusionMatrix(predict_log$data$response, testData$default)
confusionMatrix(predict_log_th$data$response, testData$default)
confusionMatrix(predict_log_th2$data$response, testData$default)

# Plot ROC
library(pROC)
library(ggplot2)

roc_curve1 <- roc(as.numeric(testData$default), as.numeric(predict_log$data$response), curve=TRUE)
round(auc(roc_curve1),3)

plot(roc_curve1)
ggroc(roc_curve1) 

roc_curve2 <- roc(as.numeric(testData$default), as.numeric(predict_log_th2$data$response), curve=TRUE)
round(auc(roc_curve2),3)

ggroc(list("Logreg" = roc_curve1, "Logreg_th" = roc_curve2)) +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  labs(title = "Logistic regression ROC Curves Comparison", x = "1 - Specificity", y = "Sensitivity") +
  theme(legend.position = "bottom")

###################### Decision Tree ########################
# hyper parameter tuning
lrn = makeLearner("classif.rpart", predict.type = "prob")
lrn
set.seed(123456789)
ps = makeParamSet(
  makeDiscreteParam("cp", seq(0.0001, 0.02,0.001)),
  makeDiscreteParam("minsplit", seq(1, 50,1))
)
ctrl = makeTuneControlGrid()

library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())
parallelStop()
parallelStartSocket(8)
tune.res = tuneParams(lrn, credit.task, resampling = rin, par.set = ps,
                      measures = list(credit.costs, mmce), control = ctrl, show.info = FALSE)
tune.res
as.data.frame(tune.res$opt.path)[1:3]


# threshold = 0.5
lrn1 = makeLearner("classif.rpart", cp=0.0011, minsplit=4, predict.type = "prob")
mod1 = train(lrn1, credit.task)
pred1 = predict(mod1, task = credit.task)
pred1

th=costs[2,1]/(costs[2,1] + costs[1,2])
th
pred.th = setThreshold(pred1, th)
pred.th


# Cross-validated performance with theoretical thresholds
rin = makeResampleInstance("CV", iters = 3, task = credit.task)
lrn = makeLearner("classif.rpart", predict.type = "prob", predict.threshold = th)
r = resample(lrn, credit.task, resampling = rin, measures = list(credit.costs, mmce), show.info = FALSE)
r

d = generateThreshVsPerfData(r, measures = list(credit.costs, mmce))
plotThreshVsPerf(d, mark.th = th)

# Empirical Threshold
tune.res = tuneThreshold(pred = r$pred, measure = credit.costs)
tune.res

tune.res2 = tuneThreshold(pred = r$pred, measure = mmce)
tune.res2

th2 = 0.6181001
pred.th2 = setThreshold(pred, th2)
pred.th2

# Cross-validated performance with default thresholds
performance(setThreshold(r$pred, 0.5), measures = list(credit.costs, mmce))
performance(pred.th2, measures = list(credit.costs, mmce))

# display confusion matrix
predict_dt = predict(mod, newdata = testData)
predict_dt_th = setThreshold(predict_dt, th)
predict_dt_th2 = setThreshold(predict_dt, th2)


confusionMatrix(predict_dt$data$response, testData$default)
confusionMatrix(predict_dt_th$data$response, testData$default)
confusionMatrix(predict_dt_th2$data$response, testData$default)

# PLOT AUC
library(pROC)
library(ggplot2)

roc_curve1 <- roc(as.numeric(testData$default), as.numeric(predict_dt$data$response), curve=TRUE)
round(auc(roc_curve1),3)

plot(roc_curve1)
ggroc(roc_curve1) 

roc_curve2 <- roc(as.numeric(testData$default), as.numeric(predict_dt_th$data$response), curve=TRUE)
round(auc(roc_curve2),3)

ggroc(list("DT" = roc_curve1, "DT_th" = roc_curve2)) +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  labs(title = "DT ROC Curves Comparison", x = "1 - Specificity", y = "Sensitivity") +
  theme(legend.position = "bottom")

########################### XGBoost ######################
# hyper parameter tuning
library(xgboost)
library(Matrix)
cv_index <- createDataPartition(trainData$default, p = 0.7, list = FALSE)
CV.trainData <- trainData[-cv_index, ]
nrow(CV.trainData)/ nrow(trainData)

# XGBoost requires numeric data for features
train_matrix <- model.matrix(default ~ . - 1, data = CV.trainData)  # One-hot encoding
train_labels <- as.numeric(CV.trainData$default) - 1  # Convert to 0 and 1

# Create DMatrix objects for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)

#create tasks
traintask <- makeClassifTask(data = CV.trainData,target = "default")

#do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list(objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet(makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
mytune

lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#create tasks
traintask2 <- makeClassifTask(data = trainData,target = "default",
                              weights = weights)


#do one hot encoding`<br/> 
traintask2 <- createDummyFeatures (obj = traintask2) 
#testtask <- createDummyFeatures (obj = testtask)

xgmodel <- train(learner = lrn_tune, traintask2)
##### threshold = 0.5 by default
predict_xgb <- predict(xgmodel,traintask2)

predict_xgb_th = setThreshold(predict_xgb, th)
pred.th

# Performance with default thresholds 0.5
performance(predict_xgb, measures = list(credit.costs, mmce))

# Performance with default thresholds 0.31
performance(predict_xgb_th, measures = list(credit.costs, mmce))

# Cross-validated performance with theoretical thresholds
parallelStop()
parallelStartSocket(4)
rin = makeResampleInstance("CV", iters = 3, task = traintask2)
configureMlr(on.par.without.desc = "quiet")
lrn2 <- setHyperPars(lrn_tune,
                     predict.threshold = th)
set.seed(123)

r = resample(lrn2,
             traintask2,
             resampling = rin,
             measures = list(credit.costs, mmce),
             show.info = FALSE)
r

d = generateThreshVsPerfData(r, measures = list(credit.costs, mmce))
plotThreshVsPerf(d, mark.th = th)

##Empirical Thresholding
# Tune the threshold based on the predicted probabilities on the 3 test data sets
tune.res = tuneThreshold(pred = r$pred, measure = credit.costs)
tune.res

tune.res2 = tuneThreshold(pred = r$pred, measure = mmce)
tune.res2

th2 = 0.6968372

predict_xgb_th2 = setThreshold(predict_xgb, th2)
predict_xgb_th2

performance(predict_xgb, measures = list(credit.costs, mmce))

performance(predict_xgb_th, measures = list(credit.costs, mmce))

performance(predict_xgb_th2, measures = list(credit.costs, mmce))

# display confusion matrix
testtask <- makeClassifTask(data = testData,target = "default")
testtask <- createDummyFeatures(obj = testtask)
predict_xgbmodel = predict(xgmodel, testtask)
predict_xgbmodel_th2 = setThreshold(predict_xgbmodel, th2)

confusionMatrix(predict_xgbmodel$data$response,predict_xgbmodel$data$truth)
confusionMatrix(predict_xgbmodel_th2$data$response,predict_xgbmodel_th2$data$truth)

# PLOT AUC
roc_curve1 <- roc(as.numeric(testData$default), as.numeric(predict_xgbmodel$data$response), curve=TRUE)
round(auc(roc_curve1),3)

plot(roc_curve1)
ggroc(roc_curve1) 

roc_curve2 <- roc(as.numeric(testData$default), as.numeric(predict_xgbmodel_th2$data$response), curve=TRUE)
round(auc(roc_curve2),3)

ggroc(list("XGBoost" = roc_curve1, "XGBoost_th" = roc_curve2)) +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  labs(title = "XGBoost ROC Curves Comparison", x = "1 - Specificity", y = "Sensitivity") +
  theme(legend.position = "bottom")