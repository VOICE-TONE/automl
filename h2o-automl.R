## Testing H20 & WEKA AutoML script
library(dplyr)
library(ggplot2)
library(tidyr)
library(h2o)
library(RWeka)
library(forcats)

h2o.init()

### Set the seed to be able to repeate the same sample
set.seed(9010)

### Splitting dates into year, month and days
bia <- bia %>% separate(col=BOOKED_DATE,into = c("BOOKED_YEAR","BOOKED_MONTH", "BOOKED_DAY"))

bia1 <- bia %>% separate(col=LEAD_DATE,into = c("LEAD_YEAR","LEAD_MONTH", "LEAD_DAY"))

bia1 <- bia1 %>% separate(col=REQUEST_DATE,into = c("REQUEST_YEAR","REQUEST_MONTH", "REQUEST_DAY"))

bia1 <- bia1 %>% select(-c(TRAVEL_REQUEST_UUID,DEPARTURE_DATE, ARRIVAL_DATE,REQUEST_YEAR,LEAD_YEAR,BOOKED_YEAR,VISIT_UUID,MARKET,CUSTOMER_UUID,SALES_AGENT_UUID))

### counting NA and verify features types 

colSums(is.na(bia1))

apply(bia1, 2, typeof)


### Encoding features and converting to factor


bia1 <- apply(bia1, 2, fct_explicit_na) %>% as.data.frame()

bia1 <- apply(bia1, 2, as.factor) %>% as.data.frame()


bia1$BOOKING_VALUE <- as.numeric(as.character(bia1$BOOKING_VALUE))

bia1$BOOKING_VALUE[is.na(bia1$BOOKING_VALUE)] <- c(0)


### counting NA and verify features types after encoding

colSums(is.na(bia1))


### Splitting the dataset into train and test

row_indices <- sample(1:nrow(bia1))

smp_size <- floor(0.75*length(row_indices))

train_bia <- bia1[row_indices[1:smp_size],]

test_bia <- bia1[row_indices[(smp_size+1):length(row_indices)],]



### converting R dataframe in to H2O dataframe

train_bia_h2o <- as.h2o(train_bia)

test_bia_h2o <- as.h2o(test_bia)

### Descriptive stats

h2o.describe(train_bia_h2o)

h2o.describe(test_bia_h2o)


### AutoML

response="OPPORTUNITY_STAGE"

predictors=setdiff(names(bia1), c(response))


aml <- h2o.automl(y=response, x=predictors, training_frame =train_bia_h2o, max_models = 10, seed = 1)

### Leaderboard

lb <- aml@leaderboard

print(lb)

print(lb, n=nrow(lb))

### Get IDS for all models

model_ids <- as.data.frame(aml@leaderboard$model_id)[,1]

### Get the "All models" stacked Ensemble model

se <- h2o.getModel(grep("StackedEnsemble_AllModels", model_ids, value = TRUE)[1])

### Get the stacked Ensemble metalearner model

metalearner <- h2o.getModel(se@model$metalearner$name)

### Examine variable importance

h2o.varimp_plot(metalearner)

### Visualizing


install.packages("autoweka")






