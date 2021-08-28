# Can Political Inclinations Be Reliably Predicted Using Profile Pictures?

## Main Contributors are 
    - Babandeep Singh (babandeep193@gmail.com)
    - Theja Tulabandhula (tt@theja.org)

## Paper Summary
Recent developments in facial recognition technology have enabled the development of algorithms that detect personality and behavioral traits from imagery. Recently, these learning capacities have been extended to detect political inclinations from display pictures (DP) present on social networking (e.g., Facebook) and dating websites. In this paper, we make an attempt to further understand the relationship between facial features and political preferences. In particular, we interpret how this relationship is influenced by self-reported covariates such as gender and country, as well as estimated covariates such as ethnicity. Our results indicate that self-reported and estimated covariates do influence the accuracy of prediction, improving image only prediction performance (measured using area under the ROC curve) by 2-10% on the absolute scale. 

## Project Pipeline

The repository contains results_for_paper which contains data_for_paper which is the insightful data about the data, and results_eda which contains the trained models results for the 3 segments we experimented : whole dataset, (country, gender and source) segment and (country, gender, source and estimated ethnicity). The estimated ethnicity is from another black box model mentioned in the paper. 

## GitHub walk through

1. Environment Setup 
    - Since this project is a collaboration we relied on Google Colab for most of the basic programming which has capabilities of python 3.7 version.
    
2. Analysis :
    - [analysis.ipunb](https://github.com/thejat/facial-political-recognition/blob/master/analysis.ipynb) - This notebook reads the data and checks for the missing values, variable importance of the categorical features. 
    - [data_analysis_paper.ipynb](https://github.com/thejat/facial-political-recognition/blob/master/data_analysis_paper.ipynb) - This notebook reads the data from the data_for_paper and gives the insights via count plots and sample sizes for different segments.
    - [results_eda.ipynb]() - This notebook gives the insights in the model and segment performace metrics to understand the underlying petterns and influence of vatious covariates from self reported to estimated. 

3. Experiments :
    - [exp_whole_dataset.ipynb](https://github.com/thejat/facial-political-recognition/blob/master/exp_whole_dataset.ipynb) - Notebook to experiment on the whole dataset and saves the results in the csv format.
    - [segment_sr.ipynb](https://github.com/thejat/facial-political-recognition/blob/master/segment_sr.ipynb) - Notebook for experiment with segmented population based on (country, gender, source). 
    - [performance_vs_ethnicity.ipynb](https://github.com/thejat/facial-political-recognition/blob/master/performance_vs_ethnicity.ipynb) - Notebook for experiment with a further breakdown at estimated ethnicity groups. 
    
## Conclusion of the study 
In this work, we investigated whether self-reported covariates can have an impact in predicting the political inclination of individuals using their profile pictures. We observe that some self-reported covariates such as country, gender do have a positive influence on the predictability of political inclination, whereas others such as age do not. Further, estimated features (estimated from profile pictures using pre-trained black box models) enable a modest bump in performance (as measured by AUC and accuracy) as well. The prediction performances also vary across model classes (two of which were used in this work). The variation of performance when data is segmented based on self-reported covariates can also likely be attributed to the quality of the input features (image descriptors and estimated features) and the representation levels of liberals and conservatives in various segments. 

    
