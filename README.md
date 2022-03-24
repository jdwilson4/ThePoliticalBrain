# ThePoliticalBrain
This repository provides public access to the code, analysis, and anonymized data associated with the political brain study in the manuscript "Functional Connectivity Signatures of Political Ideology" by Seo-Eun Yang, James D. Wilson (me), Zhong-Lin Lu, and Skyler Cranmer. Please send comments or questions to me at wilsonj41@upmc.edu or jdwilson4@usfca.edu.


## Accessing the Data 
Anonymized functional connectivity data for the 174 participants in the Wellbeing data set is available at this link: 

https://drive.google.com/file/d/1cI9ngWViE4TvO2wS7DwYMIKuKE8lJXXB/view?usp=sharing 

This will bring you to a Google Drive that requires permission to download. Please email me at wilsonj41@upmc.edu or jdwilson4@usfca.edu to request access. When you request access, please let me know the purpose of download. If you use the data for your own research, please cite the following papers: 

Seo-Eun Yang, James D. Wilson, Zhong-Lin Lu, and Skyler Cranmer. Functional Connectivity Signatures of Political Ideology *DOI to come*

and

Garren Gaut, Brandon Turner, Zhong-Lin Lu, Xiangrui Li, William A Cunningham, and Mark Steyvers. Predicting task and subject differences with functional connectivity and blood-oxygen-level-dependent variability. *Brain
connectivity*, 9(6):451–463, 201. 

The data from the above link is a `mat` file called `tc.mat`. For your own use, this file can be opened directly using *Matlab*. Running BrainNetCNN below will require that you have `tc.mat` downloaded.

## Running BrainNetCNN (Optional)



## Analyzing the Results

The following will implement the steps we took to analyze the features obtained by BrainNetCNN for the task fMRI data.

```{r} 
trial
```
