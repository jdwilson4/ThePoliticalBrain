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


## Analyzing the Results

The following will implement the steps we took to analyze the features obtained by BrainNetCNN for the task fMRI data and to obtain the figures and tables in the Yang et al. manuscript.

The following were performed in RStudio Version 1.4.1717. 

**Loading the Features from BrainNetCNN** 

Here, we will directly load the data `fMRI_Task_Features.csv` available in this repository. Alternatively, you can load the **placeholder** file from running the BrainNetCNN algorithm in the above section.

First, set `directory` to be the location of the `fMRI_Task_Features.csv` file from this repository. Then run the following:

```
setwd("directory")
dat <- read.csv(file = "fMRI_Task_Features.csv", header = TRUE)
truth <- dat$conservative_you
```

**Summaries of Socio-Economic Survey Responses**

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


**Running Association Analysis of features against true ideology**

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

**Running principal component analysis (PCA) on the Predicted Political Ideology Scores**

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


