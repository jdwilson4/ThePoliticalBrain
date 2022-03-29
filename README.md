# ThePoliticalBrain
This repository provides public access to the code, analysis, and anonymized data associated with the political brain study in the manuscript **Functional Connectivity Signatures of Political Ideology** by Seo-Eun Yang, James D. Wilson , Zhong-Lin Lu, and Skyler Cranmer. Please send comments or questions to me at wilsonj41@upmc.edu or jdwilson4@usfca.edu.


## Accessing the Data 

Anonymized functional connectivity data for the 174 participants in the Wellbeing data set is available at this link: 

https://drive.google.com/file/d/1cI9ngWViE4TvO2wS7DwYMIKuKE8lJXXB/view?usp=sharing 

This will bring you to a Google Drive that requires permission to download. Please email me at wilsonj41@upmc.edu or jdwilson4@usfca.edu to request access. When you request access, please let me know the purpose of download. If you use the data for your own research, please cite the following papers: 

Seo-Eun Yang, James D. Wilson, Zhong-Lin Lu, and Skyler Cranmer. **Functional Connectivity Signatures of Political Ideology** *DOI to come, currently under revisions*

and

Garren Gaut, Brandon Turner, Zhong-Lin Lu, Xiangrui Li, William A Cunningham, and Mark Steyvers. **Predicting task and subject differences with functional connectivity and blood-oxygen-level-dependent variability**. *Brain
connectivity*, 9(6):451–463, 201. 

The data from the above link is a `mat` file called `tc.mat`. For your own use, this file can be opened directly using *Matlab*. Running BrainNetCNN below will require that you have `tc.mat` downloaded.

## Running BrainNetCNN (Optional)

This section will show you how to run the BrainNetCNN algorithm on the Wellbeing functional connectivity data set as done in the Yang, et al. paper above. Note that the BrainNetCNN extracted features from the Wellbeing dataset are available in the *fMRI_Task_Features.csv* file. We will be using this .csv file in the next section to further analyze the results. Now we will describe how to run BrainNetCNN on the raw functional connectivity data.

You will need to make sure the following dependencies are installed on Python version 2.7.5. or greater:

• numpy –1.19.2
• scipy –1.2.1
• h5py –2.10.0
• matplotlib –3.3.4 
• pickle –0.7.5
• Caffe –1.0.0 
• csv –1.0
• os
• sys
• pandas –0.25.0 
• imblearn –0.5.0 
• ann4brains
• sklearn –0.21.3 
• itertools
• random

We modified the original BrainNetCNN and deconvolutional network architecture for our own purpose. The original BrainNetCNN code is originally from https://github.com/jeremykawahara/ann4brains. The original deconvolutional network architecture code is originally from https://shengshuyang.github.io/A-step-by-step-guide-to-Caffe.html and https://github.com/sukritshankar/CNN-Saliency-Map/blob/master/find_saliency_map.py.

Make sure that the `tc.mat` file and the `Running BrainNetCNN` are both downloaded and that the `tc.mat` file is located in the `Running BrainNetCNN` folder. Now you can simply run the following code from the terminal:

``` 
python main.py
```

This will output two files: **placeholder** and **placeholder** 


## Analyzing the results

The following will implement the steps we took to analyze the features obtained by BrainNetCNN for the task fMRI data and to obtain the figures and tables in the Yang et al. manuscript. To reproduce the analysis performed in the manuscript, run Steps **I. - VI.** below.

The following were performed in RStudio Version 1.4.1717. 

### I. Loading the features from BrainNetCNN 

Here, we will directly load the data `fMRI_Task_Features.csv` available in this repository. Alternatively, you can load the **placeholder** file from running the BrainNetCNN algorithm in the above section.

First, set `directory` to be the location of the `fMRI_Task_Features.csv` file from this repository. Then run the following:

```
setwd("directory")
dat <- read.csv(file = "fMRI_Task_Features.csv", header = TRUE)
truth <- dat$conservative_you
```

### II. Summaries of socio-economic survey responses

This code provides the summary of the survey responses of the participants in the Wellbeing study and provides the information in **Table 2** of the manuscript.

```
# organize data to be analyzed

data.x <- data.frame(Age = dat$age, Religious = dat$HowReligious, Educ = dat$education1_you, 
                     Educ_father = dat$education_father, Educ_mother = dat$education_mother,
                     Grewup = dat$cityGrewupConservative, Now = dat$cityNowConservative, 
                     Cons_father = dat$conservative_father, Cons_mother = dat$conservative_mother,
                     Income = dat$income_you, Income_parent = dat$income_parent, Male = dat$isMale,
                     Affect = dat$Affect, Empathy = dat$Empathy, Encoding = dat$Encoding, 
                     GoNogo = dat$GoNogo, Resting = dat$Resting, Retrieval = dat$Retrieval, 
                     Reward = dat$Reward, ToM = dat$ToM, Working_Mem = dat$WorkingMem)


# descriptive summaries of Table 2
table(data.x$Male)
mean(data.x$Age)
sd(data.x$Age)
summary(data.x$Age)
table(data.x$Educ)
table(data.x$Educ_father)
table(data.x$Educ_mother)
table(data.x$Cons_father)
table(data.x$Cons_mother)
table(data.x$Income)
table(data.x$Income_parent)
table(data.x$Religious)
table(data.x$Now)
table(data.x$Grewup)
table(truth)

# correlations and p-values of Table 2
cor.test(data.x$Male, truth)
cor.test(data.x$Age, truth)
cor.test(data.x$Educ, truth)
cor.test(data.x$Educ_father, truth)
cor.test(data.x$Educ_mother, truth)
cor.test(data.x$Cons_father, truth)
cor.test(data.x$Cons_mother, truth)
cor.test(data.x$Income, truth)
cor.test(data.x$Income_parent, truth)
cor.test(data.x$Religious, truth)
cor.test(data.x$Now, truth)
cor.test(data.x$Grewup, truth)
```


### III. Running association analysis of features against true ideology

This code provides plot **Figure 1** -- pairwise scatterplots that show associations among the features (political scores from each task) as well as associations between the predicted political scores and the true political ideology of each participant.

To create this plot, you'll need the `GGally` and `ggplot2` packages installed and loaded in R.

``` 
# load the needed packages
install.packages("ggplot2")
install.packages("GGally")
library(ggplot2)
library(GGally)

# get a .pdf of the plot using the ggpairs function
pdf("Associations_Figure1.pdf", width = 16, height = 16)
ggpairs(data.frame(Ideology = truth, Affect = dat$Affect, Empathy = dat$Empathy, Reward = dat$Reward, Retrieval = dat$Retrieval, Resting = dat$Resting, 
                   GoNoGo = dat$GoNogo, Encoding = dat$Encoding, ToM = dat$ToM, WorkingMem= dat$WorkingMem, Extremity = as.factor(Extremity)), 
                   mapping =  aes(color = Extremity), alpha = 0.5, upper = list(continuous = wrap("cor"))) + theme_grey(base_size = 15)

dev.off()
```

### IV. Running principal component analysis (PCA) on the predicted political ideology scores

This first chunk of code provides plot **Figure 2** -- the scree plot and biplot of the top 2 principal components, with scores colored by ideology. 

To get this plot, you'll need the `factoextra` packages installed and loaded in R. 

```
# load the needed package
install.packages("factoextra")
library(factoextra)

# get the scores matrix
scores_matrix <- as.matrix(data.frame(Affect = dat$Affect, Empathy = dat$Empathy, Reward = dat$Reward, Retrieval = dat$Retrieval, Resting = dat$Resting, 
                            GoNoGo = dat$GoNogo, Encoding = dat$Encoding, ToM = dat$ToM, WorkingMem= dat$WorkingMem))

# run pca and visualize the scree plot
pcs <- prcomp(scores_matrix, scale = TRUE)

# set node colors as "Liberal" vs. "Conservative"
groups <- rep("Conservative", dim(dat)[1])
groups[which(truth == 6)] <- "Conservative"
groups[which(truth == 1)] <- "Liberal"
groups[which(truth == 2 | truth == 3)] <- "Liberal"
groups <- as.factor(groups)


# biplot with each party type - the right panel of Figure 2
pdf("Biplot_Right_Figure2.pdf", width = 10, height = 7.5)
fviz_pca_ind(pcs,
             col.ind = groups, # color by groups
             addEllipses = TRUE, # Concentration ellipses
             ellipse.type = "confidence",
             legend.title = "Ideology",
             repel = TRUE
) + theme_gray(base_size = 15)

dev.off()

# scree plot - the left panel of Figure 2
pdf("Scree_plot_Left_Figure2.pdf", width = 6, height = 7.5)
fviz_eig(pcs) + theme_gray(base_size = 15)

dev.off()
```


This next chunk generates **Table 3** of the manuscript -- the contribution of each task to the top 5 principal components of the predicted political ideology score matrix.

```
# variable contributions to the pcs
var_coord_func <- function(loadings, comp.sdev){
  loadings*comp.sdev
}

# compute coordinates
loadings <- pcs$rotation
sdev <- pcs$sdev
var.coord <- t(apply(loadings, 1, var_coord_func, sdev)) 

var.cos2 <- var.coord^2

comp.cos2 <- apply(var.cos2, 2, sum)
contrib <- function(var.cos2, comp.cos2){var.cos2*100/comp.cos2}
var.contrib <- t(apply(var.cos2,1, contrib, comp.cos2))

# print Table 3 -- the first 5 pcs
Table3 <- var.contrib[, 1:5]

```

### V. Predicting political ideology with BrainNetCNN features

This code chunk generates the data and plot for **Figure 3** of the manuscript. In particular, this will plot the AUCs and report the accuracies of each model considered in the manuscript. Averages and standard deviations of each metric are reported based on Monte Carlo cross validation over 1000 randomly selected samples from the full population.

This first chunk creates the AUC plot in the right panel of **Figure 3**. 


```
# load the needed packages
install.packages("pROC")
install.packages("verification")
library(pROC)
library(verification)

# binarize political ideology into "Conservative" and "Liberal" (0 vs. 1)
truth1 <- truth 
truth1[truth1 < 4] = 0
truth1[truth1 > 3] = 1

# calculate AUCs for each model
aucs <- list()

aucs[[1]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ dat$Affect, family = "binomial"), type = "response"), CI = TRUE)$A.boot
aucs[[2]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ Empathy, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot 
aucs[[3]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ Encoding, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot
aucs[[4]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ GoNogo, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot 
aucs[[5]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ Resting, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot
aucs[[6]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ Retrieval, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot
aucs[[7]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ Reward, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot
aucs[[8]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ ToM, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot
aucs[[9]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ WorkingMem, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot 

# all tasks
aucs[[10]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ Affect + Empathy + Encoding + GoNogo + Resting + Retrieval + Reward +
                                                         ToM + WorkingMem, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot
# parent conservatism only
aucs[[11]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ conservative_father + conservative_mother, data = dat, family = "binomial"), 
                      type = "response"), CI = TRUE)$A.boot

# parent conservatism + all tasks
aucs[[12]] <-  roc.plot(as.numeric(truth1), predict(glm(truth1 ~ conservative_father + conservative_mother + Affect + Empathy + Encoding + GoNogo + Resting + Retrieval + Reward + ToM + WorkingMem, data = dat, family = "binomial"), type = "response"), CI = TRUE)$A.boot

# all survey only
aucs[[13]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ age + education1_you +
                                                         education_father + education_mother + cityGrewupConservative +
                                                         cityNowConservative + conservative_father + conservative_mother +
                                                         income_you + income_parent + isMale, data = dat, family = "binomial"), type = "response"), 
                                                         CI = TRUE)$A.boot

# survey + tasks
aucs[[14]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ age + education1_you +
                                                         education_father + education_mother + cityGrewupConservative +
                                                         cityNowConservative + conservative_father + conservative_mother +
                                                         income_you + income_parent + isMale + Affect + Empathy + Encoding + GoNogo + Resting + Retrieval 
                                                         + Reward + ToM + WorkingMem, data = dat, family = "binomial"), type = "response"), 
                                                         CI = TRUE)$A.boot 

# survey without parent conservatism
aucs[[15]] <- roc.plot(as.numeric(truth1), predict(glm(truth1 ~ age + education1_you +
                                                         education_father + education_mother + cityGrewupConservative +
                                                         cityNowConservative + income_you + income_parent + isMale, data = dat, family = "binomial"), 
                                                         type = "response"), CI = TRUE)$A.boot 


###Means and standard deviation. 
mean_auc <- rep(0, 15)
sd_auc <- rep(0, 15)
for(i in 1:15){
  mean_auc[i] <- mean(aucs[[i]])
  sd_auc[i] <- sd(aucs[[i]])
}

#A function to add arrows on the chart
error.bar <- function(x, y, upper, lower=upper, length=0.1,...){
  arrows(x,y+upper, x, y-lower, angle=90, code=3, length=length, ...)
}

x <- order(mean_auc)

# Creating Figure 3 (AUC figure)

dev.new()
my.bar <- barplot(mean_auc[x], 
                  names.arg = c("Affect", "Empathy", "Encoding", "GoNogo", "Resting", "Retrieval", "Reward", "ToM", "WorkingMem", "All Tasks", 
                                "Parental Conservatism", "Parental Cons. + All Tasks", 
                                "All Survey", "All Survey + All Tasks", "Survey w/o Parent Cons.")[x],
                  xaxt = "n", ylab = "Predictive AUC", ylim = c(0,1), col = c(rep("orange", 5), rep("grey", 7), "darkblue", "darkblue", "darkblue"),
                  density = 30, angle = 36)

error.bar(my.bar, mean_auc[x],sd_auc[x])

labs <- c("Affect", "Empathy", "Encoding", "GoNogo", "Resting", "Retrieval", "Reward", "ToM", "WorkingMem", "All FC Tasks", 
          "Parent", "All Survey", "Parent + FC",  "Survey + FC", "Survey w/o Parent Cons.")[x]

text(cex=.85, x = my.bar, y=-0.15, labs, xpd=TRUE, srt=90)
```


This next chunk calculates the accuracies for each model considered in the study as provided in the left panel of **Figure 3**. To obtain predictive accuracies, we use the `caret` and `e1071` packages in R.

```
install.packages("e1071")
library(caret)
library(e1071)

train_control <- trainControl(method = "LGOCV", number = 1000)

dat <- cbind(dat, truth1 = as.factor(truth1))


##############################Comparing models via predictive accuracies#########################
model <- list()
model[[1]] <- train(truth1 ~ Affect, data = dat, method = "glm", trControl = train_control)
model[[2]] <- train(truth1 ~ Empathy, data = dat, method = "glm", trControl = train_control)
model[[3]] <- train(truth1 ~ Encoding, data = dat, method = "glm", trControl = train_control)
model[[4]] <- train(truth1 ~ GoNogo, data = dat, method = "glm", trControl = train_control)
model[[5]] <- train(truth1 ~ Resting, data = dat, method = "glm", trControl = train_control)
model[[6]] <- train(truth1 ~ Retrieval, data = dat, method = "glm", trControl = train_control)
model[[7]] <- train(truth1 ~ Reward, data = dat, method = "glm", trControl = train_control)
model[[8]] <- train(truth1 ~ ToM, data = dat, method = "glm", trControl = train_control)
model[[9]] <- train(truth1 ~ WorkingMem, data = dat, method = "glm", trControl = train_control)

#all tasks
model[[10]] <- train(truth1 ~ Affect + Empathy + Encoding + GoNogo + Resting + Retrieval + Reward +
                       ToM + WorkingMem, data = dat, method = "glm", trControl = train_control)

#parent conservatism only
model[[11]] <- train(truth1 ~ conservative_father + conservative_mother, data = dat, method = "glm", trControl = train_control)

#parent conservatism + tasks
model[[12]] <- train(truth1 ~ conservative_father + conservative_mother + Affect + Empathy + Encoding + GoNogo + Resting + Retrieval + Reward +
                       ToM + WorkingMem, data = dat, method = "glm", trControl = train_control)
#all survey only
model[[13]] <- train(truth1 ~  age + HowReligious + education1_you +
                       education_father + education_mother + cityGrewupConservative +
                       cityNowConservative + conservative_father + conservative_mother +
                       income_you + income_parent + isMale, data = dat, method = "glm", trControl = train_control)
#all survey + tasks
model[[14]] <- train(truth1 ~  age + HowReligious + education1_you +
                       education_father + education_mother + cityGrewupConservative +
                       cityNowConservative + conservative_father + conservative_mother +
                       income_you + income_parent + isMale + Affect + Empathy + Encoding + GoNogo + Resting + Retrieval + Reward +
                       ToM + WorkingMem, data = dat, method = "glm", trControl = train_control)

#suvey w/o parent conservatism
model[[15]] <- train(truth1 ~  age + HowReligious + education1_you +
                       education_father + education_mother + cityGrewupConservative +
                       cityNowConservative +
                       income_you + income_parent + isMale, data = dat, method = "glm", trControl = train_control)


accuracies <- rep(0, 15)
st.devs <- rep(0, 15)

for(i in 1:length(model)){
  accuracies[i] <- model[[i]]$results$Accuracy[which.max(model[[i]]$results$Accuracy)]
  st.devs[i] <- model[[i]]$results$AccuracySD[which.max(model[[i]]$results$Accuracy)]
}

# accuracy table on the left of Figure 3.
accuracy_table <- data.frame(Model = c("Affect", "Empathy", "Encoding", "Gonogo", "Resting",
                                       "Retrieval", "Reward", "ToM", "WorkingMem",
                                       "All Tasks", "Parent Cons only", "Parent Cons + Tasks", 
                                       "All Survey", "All Survey + Tasks", "Survey w/o Parent Cons"),
                             Mean = accuracies, SD = st.devs)

```

### VI. Variable importance of BrainNetCNN and survey-based features

This code chunk will reproduce the variable importance summaries provided in **Table 4** of the manuscript.

To run this chunk, you will need the `glmnet` package installed and loaded in R.

```
# load the needed package
install.packages("glmnet")
library(glmnet)

# run LASSO on full model
reg_lasso <- cv.glmnet(x = as.matrix(data.x),
                      y = truth1, family = "binomial")

best.lambda <- reg_lasso$lambda.min
# first column of Table 4
coef(reg_lasso, s = "lambda.min")

# now get the coefficients and standard errors from the covariates that were had importance > 0 from above.
regression_results <- glm(truth1 ~ Reward + Retrieval + Empathy + Cons_mother + Cons_father + Educ_father + Educ_mother,
                          data = data.x)
                          
# last two columns of Table 4
summary(regression_results)

```
