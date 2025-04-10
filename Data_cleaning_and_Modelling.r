# data cleaning, data exploration, feature engineering, dummy variable creation
rm(list = ls())

library(ggplot2)
library(dplyr)

###### read lending club data from csv file 
loans <- read.csv('lending_club_loans.csv', stringsAsFactors = FALSE)


str(loans) 
table(as.factor(loans$loan_status))
39786/42535

######  create default target variable
summary(as.factor(loans$loan_status))
loans$default <- case_when(
  loans$loan_status %in% c("Charged Off","Default","Does not meet the credit policy. Status:Charged Off","In Grace Period","Late (16-30 days)","Late (31-120 days)") ~ 1,
  TRUE ~ 0)
summary(as.factor(loans$default))
nrow(loans)
(6448/42535)*100 #15.15% defaulters

table(loans$meet_CreditPolicy, loans$default)
(761/(761+1988))*100
(5687/(5687+34099))*100
# do not meet CP default rate = 27.68%
# meets CP default rate = 14.29%
summary(as.factor(loans$meet_CreditPolicy))
finaldata <- loans %>%
  select("annual_inc",
         "delinq_2yrs",
         "dti",
         "emp_length",
         "fico_range_low",
         "grade",
         "home_ownership",
         "installment",
         "int_rate",
         "last_fico_range_high",
         "last_fico_range_low",
         "loan_amnt",
         "loan_status",
         "open_acc",
         "out_prncp",
         "pub_rec",
         "pub_rec_bankruptcies",
         "purpose",
         "revol_bal",
         "term",
         "total_acc",
         "verification_status",
         "meet_CreditPolicy",
         "default",
         "revol_util"
          )
str(finaldata)
summary(finaldata)

##### feature engineering
# Dummy variables
datamodel <- NULL

datamodel <- finaldata
datamodel$term <- trimws(datamodel$term) 

datamodel <- datamodel %>%
  mutate(revol_util = as.numeric(sub("%","", revol_util)),
         int_rate = as.numeric(sub("%","", int_rate)),
         emp_length_less1 = case_when(emp_length %in% c("< 1 year",0) ~ 1, TRUE ~ 0),
         emp_length_less5 = case_when(emp_length %in% c("1 year","2 years","3 years","4 years","5 years") ~ 1, TRUE ~ 0),
         emp_length_less10 = case_when(emp_length %in% c("6 years","8 years","9 years","10 years") ~ 1, TRUE ~ 0),
         emp_length_more10 = case_when(emp_length %in% c("10+ years") ~ 1, TRUE ~ 0),
         emp_ntgiven = case_when(emp_length %in% c("n/a",0) ~ 1, TRUE ~ 0),
         
         purpose_car = case_when(purpose =="car" ~ 1, TRUE ~ 0),
         purpose_home = case_when(purpose %in% c("home_improvement","house") ~ 1, TRUE ~ 0),
         purpose_personal = case_when(purpose %in% c("educational","medical","moving","vacation","wedding") ~ 1, TRUE ~ 0),
         purpose_business = case_when(purpose %in% c("small_business") ~ 1, TRUE ~ 0),
         purpose_cc = case_when(purpose %in% c("credit_card") ~ 1, TRUE ~ 0),
         purpose_debtc = case_when(purpose %in% c("debt_consolidation") ~ 1, TRUE ~ 0),
         purpose_other = case_when(purpose %in% c("major_purchase","other","renewable_energy") ~ 1, TRUE ~ 0),
         
         home_own = case_when(home_ownership %in% c("OWN") ~ 1, TRUE ~ 0),
         home_mortgage = case_when(home_ownership %in% c("MORTGAGE") ~ 1, TRUE ~ 0),
         home_rent = case_when(home_ownership == "RENT" ~ 1, TRUE ~ 0),
         home_others = case_when(home_ownership %in% c("OTHER","NONE") ~ 1, TRUE ~ 0),
         
         
         verification_y = case_when(verification_status %in% c("Source Verified","Verified") ~ 1, TRUE ~ 0),
         verification_n = case_when(verification_status == "Not Verified" ~ 1, TRUE ~ 0),
         
         term_36months_else60 = case_when(term == "36 months" ~ 1, TRUE ~ 0),
         meets_cp = case_when(meet_CreditPolicy =="Y" ~ 1, TRUE ~ 0) 
  )


datamodel <- datamodel %>%
  mutate(
    grade_A = case_when(grade =="A" ~ 1, TRUE ~ 0),
    grade_B = case_when(grade =="B" ~ 1, TRUE ~ 0),
    grade_C = case_when(grade =="C" ~ 1, TRUE ~ 0),
    grade_D = case_when(grade =="D" ~ 1, TRUE ~ 0),
    grade_E = case_when(grade =="E" ~ 1, TRUE ~ 0),
    grade_F = case_when(grade =="F" ~ 1, TRUE ~ 0),
    grade_G = case_when(grade =="G" ~ 1, TRUE ~ 0)
  )

#remove character variables
str(datamodel)
datamodel$emp_length <- NULL
datamodel$grade <- NULL
datamodel$home_ownership <- NULL
datamodel$loan_status <- NULL
datamodel$purpose <- NULL
datamodel$term <- NULL
datamodel$verification_status <- NULL
datamodel$meet_CreditPolicy <- NULL

# convert to factors
datamodel$default <- as.factor(datamodel$default)
datamodel$emp_length_less1 <- as.factor(datamodel$emp_length_less1)
datamodel$emp_length_less5 <- as.factor(datamodel$emp_length_less5)
datamodel$emp_length_less10 <- as.factor(datamodel$emp_length_less10)
datamodel$emp_length_more10 <- as.factor(datamodel$emp_length_more10)
datamodel$emp_ntgiven <- as.factor(datamodel$emp_ntgiven)
datamodel$purpose_car <- as.factor(datamodel$purpose_car)
datamodel$purpose_home <- as.factor(datamodel$purpose_home)
datamodel$purpose_personal <- as.factor(datamodel$purpose_personal)
datamodel$purpose_business <- as.factor(datamodel$purpose_business)
datamodel$purpose_cc <- as.factor(datamodel$purpose_cc)
datamodel$purpose_debtc <- as.factor(datamodel$purpose_debtc)
datamodel$purpose_other <- as.factor(datamodel$purpose_other)
datamodel$home_own <- as.factor(datamodel$home_own)
datamodel$home_mortgage <- as.factor(datamodel$home_mortgage)
datamodel$home_rent <- as.factor(datamodel$home_rent)
datamodel$home_others <- as.factor(datamodel$home_others)
datamodel$verification_y <- as.factor(datamodel$verification_y)
datamodel$verification_n <- as.factor(datamodel$verification_n)
datamodel$meets_cp <- as.factor(datamodel$meets_cp)
datamodel$term_36months_else60 <- as.factor(datamodel$term_36months_else60)

datamodel$grade_A <- as.factor(datamodel$grade_A)
datamodel$grade_B <- as.factor(datamodel$grade_B)
datamodel$grade_C <- as.factor(datamodel$grade_C)
datamodel$grade_D <- as.factor(datamodel$grade_D)
datamodel$grade_E <- as.factor(datamodel$grade_E)
datamodel$grade_F <- as.factor(datamodel$grade_F)
datamodel$grade_G <- as.factor(datamodel$grade_G)

####### Handling numeric variables

# annual_inc
# remove rows (4 not having annual income)
datamodel <- datamodel %>% filter(!is.na(annual_inc))

# Calculate the 5th and 95th percentiles
lower_bound <- quantile(datamodel$annual_inc, 0.05, na.rm = TRUE)
upper_bound <- quantile(datamodel$annual_inc, 0.95, na.rm = TRUE)
df_filtered <- datamodel[datamodel$annual_inc >= lower_bound & datamodel$annual_inc <= upper_bound, ]
hist(df_filtered$annual_inc)

df_filtered$annual_inc <- log(df_filtered$annual_inc)
str(df_filtered)

# delinq_2yrs not needed
# dti not needed
# fico_range_low not needed
# int_rate not needed
# loan_amt not needed
# open_acc
lower_bound <- quantile(df_filtered$open_acc, 0.05, na.rm = TRUE)
upper_bound <- quantile(df_filtered$open_acc, 0.95, na.rm = TRUE)
df_filtered2 <- df_filtered[df_filtered$open_acc >= lower_bound & df_filtered$open_acc <= upper_bound, ]
hist(df_filtered2$open_acc)
str(df_filtered2)

# out_prncp
df_filtered3 <- df_filtered2[df_filtered2$out_prncp <= 600,]


# pub_rec 
df_filtered4 <- df_filtered3
df_filtered4$pub_rec <- round(df_filtered4$pub_rec)
df_filtered4 <- df_filtered4[df_filtered4$pub_rec <= 2,]

# pub_rec_bankruptcies
df_filtered4 <- df_filtered4[df_filtered4$pub_rec_bankruptcies <= 1 ,]

# revol_bal
df_filtered5 <- df_filtered4
df_filtered5$revol_bal <- log(df_filtered5$revol_bal)

lower_bound <- quantile(df_filtered5$revol_bal, 0.05, na.rm = TRUE)
upper_bound <- quantile(df_filtered5$revol_bal, 0.95, na.rm = TRUE)
df_filtered5 <- df_filtered5[df_filtered5$revol_bal >= lower_bound & df_filtered5$revol_bal <= upper_bound, ]

 # total_acc not needed

# revol_util
df_filtered6 <- df_filtered5[!df_filtered5$revol_util %in% c(100.5, 101.4, 106.1,106.2,106.5,108.8), ]


# removing 913 NA's from the data
df_filtered7 <- df_filtered6[!is.na(df_filtered6$default),]
df_filtered7$default <- factor(df_filtered7$default, levels = c(1,0), labels = c("bad", "good"))
summary(df_filtered7$default)


############################## DATA EXPLORATION ##############################
###### --------- Data Exploration --------------
table(df_filtered7$default)
6448/36087
4449/26012
table(loans$default, loans$grade)
#ggplot(loans, aes(x = loans$grade)) + geom_histogram(stat = "count", fill = "skyblue", color = "black") + labs(title = "Distribution of Grades", x = "Grade", y = "Count") +theme_minimal()

ggplot(loans, aes(x = grade, fill = as.factor(default))) +
  geom_bar(position = "stack", color = "black") +
  labs(title = "Distribution of Default within different Loan Grade", 
       x = "Grade", 
       y = "Count",
       fill = "Default Status") +
  theme_minimal() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon"), 
                    name = "Loan Status", 
                    labels = c("Non-Default", "Default"))


#####  loan amt distribution
Loanamountdist <- ggplot(data=loans, aes(x=loan_amnt)) + 
  geom_histogram(aes(y=..density..),
                 col='black', 
                 fill='skyblue', 
                 alpha=0.3) +
  geom_density(adjust=3) +
  labs(title = "Distribution of Default with Loan Amount", 
       x = "Loan Amount") +
  theme_minimal()

print(Loanamountdist + theme(plot.title=element_text(face="bold")) + ggtitle('Distribution of the loan amounts') +
        theme_minimal())
summary(df_filtered7$loan_amnt)

##### home ownership distribution
homedist <- ggplot(data=loans, aes(x=home_ownership, fill=default)) + 
  geom_bar(aes(y = (..count..)/sum(..count..)), position='stack', alpha=0.5) + scale_y_continuous(labels=scales::percent)

print(homedist + theme(plot.title=element_text(face="bold")) + ggtitle('Home Ownership vs Loan Default'))


ggplot(finaldata, aes(x = home_ownership, fill = as.factor(default))) +
  geom_bar(position = "stack") +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon"), 
                    name = "Loan Status", 
                    labels = c("Non-Default", "Default")) +
  labs(title = "Distribution of Home Ownership by Default Status",
       x = "Home Ownership",
       y = "Count") +
  theme_minimal()


#####  Calculate counts and percentages for each combination of home_ownership and default
counts <- loans %>% 
  filter(!home_ownership %in% c("OTHER", "NONE")) %>%
  group_by(home_ownership, default) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(home_ownership) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  ungroup()

# Create the stacked bar chart with percentage data labels
loans %>% 
  filter(!home_ownership %in% c("OTHER", "NONE")) %>% 
  ggplot(aes(x = home_ownership, fill = as.factor(default))) +
  geom_bar(position = "stack") +
  geom_text(data = counts, aes(x = home_ownership, y = count, label = sprintf("%.1f%%", percentage)), 
            position = position_stack(vjust = 0.5), color = "black", size = 3) +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon"), name = "Loan Status", labels = c("Non-Default", "Default")) +
  labs(title = "Distribution of Home Ownership by Default Status",
       x = "Home Ownership",
       y = "Count") +
  theme_minimal()

##### Loan purpose groups distribution
table(as.factor(loans$purpose), as.numeric(loans$default))
loans$purpose_group <- case_when (
  loans$purpose == "car" ~ "car",
  loans$purpose %in% c("educational","medical","moving","vacation","wedding") ~ "personal",
  loans$purpose %in% c("home_improvement","house") ~ "home",
  loans$purpose %in% c("small_business") ~ "business",
  loans$purpose %in% c("credit_card") ~ "credit card",
  loans$purpose %in% c("debt_consolidation") ~ "debt_consolidation",
  TRUE ~ "others"
)
ggplot(loans, aes(x = purpose_group, fill = as.factor(default))) +
  geom_bar(position = "stack") +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon"), name = "Default Status", labels = c("Non-Default", "Default")) +
  labs(title = "Distribution of Purpose of loan by Default Status",
       x = "purpose",
       y = "Count") +
  theme_minimal()
loans$purpose_group <- NULL

#### FICO score distribution
ggplot(loans, aes(x = fico_range_low, fill = as.factor(default))) +
  geom_histogram(binwidth = 20, position = "stack", color = "black") +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon"), name = "Default Status", labels = c("Non-Default", "Default")) +
  labs(title = "Distribution of FICO Range Low by Default Status",
       x = "FICO Range Low",
       y = "Count") +
  theme_minimal()
