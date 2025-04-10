# Instance weighing Cost sensitive technique 
######################### CSL R ###################################
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

set.seed(123)
library(caret)
train_index <- createDataPartition(df_filtered7$default, p = 0.7, list = FALSE)
trainData <- df_filtered7[train_index, ]
testData <- df_filtered7[-train_index, ]

library(mlr)
library(mlr3)
library(mlr3learners)
library(mlr3measures)
library(mlr3misc)
library(mlr3tuning)
library(mlr3verse)


credit.task <- makeClassifTask(data= trainData, target = 'default')
credit.task <- removeConstantFeatures(credit.task)
credit.task

costs = matrix(c(0,2615.14,5957.99,   0    ),2)
colnames(costs) = rownames(costs) = getTaskClassLevels(credit.task)
costs

credit.costs = makeCostMeasure(id = "credit.costs",
                               name = "Credit costs",
                               costs = costs,
                               minimize = TRUE,
                               best = 0, worst = 5957.99)
credit.costs

rin = makeResampleInstance("CV", iters = 3, task = credit.task)
w = 17

################## logistic Regression ####################
lrn1 = makeLearner("classif.logreg", predict.type="prob")
lrn1 = makeWeightedClassesWrapper(lrn1, wcw.weight = w)
lrn1
set.seed(123456789)
ps = makeParamSet(makeDiscreteParam("wcw.weight", seq(1, 20, 0.5)))
ctrl = makeTuneControlGrid()
tune.res = tuneParams(lrn1, credit.task, resampling = rin, par.set = ps,
                      measures = list(credit.costs, mmce), control = ctrl, show.info = FALSE)
tune.res
as.data.frame(tune.res$opt.path)[1:3]
# tuned weight = 3.5


mod1 = train(lrn1, credit.task)
pred1 = predict(mod1, newdata = testData)
pred1

w = 3.5
lrn2 = makeLearner("classif.logreg", predict.type="prob")
lrn2 = makeWeightedClassesWrapper(lrn2, wcw.weight = w)
lrn2

mod2 = train(lrn2, credit.task)
pred2 = predict(mod2, newdata = testData)
pred2

performance(pred1, measures = list(credit.costs, mmce))
performance(pred2, measures = list(credit.costs, mmce))

confusionMatrix(pred1$data$response, testData$default)
confusionMatrix(pred2$data$response, testData$default)

# PLOT AUC
library(pROC)
library(ggplot2)

roc_curve1 <- roc(as.numeric(testData$default), as.numeric(pred1$data$response), curve=TRUE)
round(auc(roc_curve1),3)

plot(roc_curve1)
ggroc(roc_curve1) 

roc_curve2 <- roc(as.numeric(testData$default), as.numeric(pred2$data$response), curve=TRUE)
round(auc(roc_curve2),3)

plot(roc_curve2)
ggroc(roc_curve2) 

ggroc(list("LogReg" = roc_curve1, "LogReg_w" = roc_curve2)) +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  labs(title = "LogReg ROC Curves Comparison", x = "1 - Specificity", y = "Sensitivity") +
  theme(legend.position = "bottom")

##################### Decision Trees ###########################
w = 17
lrn = makeLearner("classif.rpart", predict.type = "prob")
lrn = makeWeightedClassesWrapper(lrn, wcw.weight = w)
lrn
set.seed(123456789)
ps = makeParamSet(
  makeDiscreteParam("wcw.weight", seq(1, 20, 0.5)),
  makeDiscreteParam("cp", seq(0.0001, 0.02,0.001))
)
ctrl = makeTuneControlGrid()
tune.res = tuneParams(lrn, credit.task, resampling = rin, par.set = ps,
                      measures = list(credit.costs, mmce), control = ctrl, show.info = FALSE)
tune.res
as.data.frame(tune.res$opt.path)[1:3]


lrn3 = makeLearner("classif.rpart", cp=0.0041, predict.type = "prob")
lrn3 = makeWeightedClassesWrapper(lrn3, wcw.weight = 17)
mod3 = train(lrn3, credit.task)
pred3 = predict(mod3, newdata = testData)
pred3

w = 2.5
lrn4 = makeLearner("classif.rpart",cp = 0.0041,predict.type = "prob")
lrn4 = makeWeightedClassesWrapper(lrn4, wcw.weight = w)
lrn4
mod4 = train(lrn4, credit.task)
pred4 = predict(mod4, newdata = testData)
pred4

performance(pred3, measures = list(credit.costs, mmce))
performance(pred4, measures = list(credit.costs, mmce))

confusionMatrix(pred3$data$response, testData$default)

confusionMatrix(pred4$data$response, testData$default)

# PLOT AUC
library(pROC)
library(ggplot2)

roc_curve1 <- roc(as.numeric(testData$default), as.numeric(pred3$data$response), curve=TRUE)
round(auc(roc_curve1),3)

plot(roc_curve1)
ggroc(roc_curve1) 

roc_curve2 <- roc(as.numeric(testData$default), as.numeric(pred4$data$response), curve=TRUE)
round(auc(roc_curve2),3)

plot(roc_curve2)
ggroc(roc_curve2) 

ggroc(list("DT" = roc_curve1, "DT_w" = roc_curve2)) +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  labs(title = "DT ROC Curves Comparison", x = "1 - Specificity", y = "Sensitivity") +
  theme(legend.position = "bottom")

################# XGBoost ################################
library(xgboost)
library(Matrix)

# XGBoost requires numeric data for features
train_matrix <- model.matrix(default ~ . - 1, data = trainData)  # One-hot encoding
train_labels <- as.numeric(trainData$default) - 1  # Convert to 0 and 1

# Create DMatrix objects for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)

#create tasks
traintask <- makeClassifTask(data = trainData,target = "default")

#do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 

#create learner
w = 17
lrn5 = makeLearner("classif.xgboost",predict.type = "prob")
lrn5 = makeWeightedClassesWrapper(lrn5, wcw.weight = w)
lrn5
lrn5$par.vals <- list(objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet(
  makeDiscreteParam("booster",values = c("gbtree","gblinear")),
  makeIntegerParam("max_depth",lower = 3L,upper = 10L),
  makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
  makeNumericParam("subsample",lower = 0.5,upper = 1),
  makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
  makeDiscreteParam("wcw.weight", seq(1, 20, 0.5))
)

rdesc <- makeResampleDesc("CV",stratify = T,iters=3L)
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#Hyper parameter tuning
mytune <- tuneParams(learner = lrn5,
                     task = traintask,
                     resampling = rdesc,
                     measures = list(credit.costs, mmce),
                     par.set = params,
                     control = ctrl,
                     show.info = T)
mytune$y

mytune

as.data.frame(mytune$opt.path)[1:8]

#set hyperparameters
lrn_tuned <- setHyperPars(lrn5,par.vals = mytune$x)

#do one hot encoding`<br/>
traintask <- credit.task
testtask <- makeClassifTask(data = testData,target = "default") 
traintask <- createDummyFeatures (obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)


xgmodel <- train(lrn5, traintask)
pred5 <- predict(xgmodel,testtask)

xgmodel_tw <- train(lrn_tuned,traintask)
pred6 <- predict(xgmodel_tw,testtask)

performance(pred5, measures = list(credit.costs, mmce))
performance(pred6, measures = list(credit.costs, mmce))

confusionMatrix(pred5$data$response, testData$default)
confusionMatrix(pred6$data$response, testData$default)

# PLOT AUC
roc_curve1 <- roc(as.numeric(testData$default), as.numeric(pred5$data$response), curve=TRUE)
round(auc(roc_curve1),3)

plot(roc_curve1)
ggroc(roc_curve1) 

roc_curve2 <- roc(as.numeric(testData$default), as.numeric(pred6$data$response), curve=TRUE)
round(auc(roc_curve2),3)

plot(roc_curve2)
ggroc(roc_curve2) 

ggroc(list("XGBoost" = roc_curve1, "XGBoost_w" = roc_curve2)) +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  labs(title = "XGBoost ROC Curves Comparison", x = "1 - Specificity", y = "Sensitivity") +
  theme(legend.position = "bottom")