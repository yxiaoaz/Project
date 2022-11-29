## Baseline Model Sharing ##

Our group is also sharing the results of Bag-of-Word models, since we probably will all use these results for our own experiments anyway. We split the dataset with 80:20 train-test ratio, and adopted oversampling (SMOTE) technique on the train set to address the imbalance issue (~150:1, NF:F).

The results are show below:

```
------------------Multinomial Naive Bayes-------------------
Starts pipeline
-------Best Model Parameters:--------
{'alpha': 0.01}
-------------------------------------
Accuracy Score : 0.93974175035868
Precision Score : 0.05813953488372093
Recall Score : 0.625
F1 Score : 0.10638297872340426
```

```
------------------BernoulliNB-------------------
Starts pipeline
-------Best Model Parameters:--------
{'alpha': 0.01}
-------------------------------------
Accuracy Score : 0.9681492109038737
Precision Score : 0.030927835051546393
Recall Score : 0.15
F1 Score : 0.05128205128205128
```

'''
Logistic Regression
-------Best Model Parameters:--------
{'C': 10000.0, 'penalty': 'l2'}
-------------------------------------
Accuracy Score : 0.9925394548063128
Precision Score : 0.3235294117647059
Recall Score : 0.275
F1 Score : 0.2972972972972973
'''


## Data Processing Part of UROP2022 Fall
**The raw txt files**

Due to the huge size of the raw .txt file bundle, the ```HKEX Reports``` folder can be found in this [link](https://drive.google.com/drive/folders/1-gYNQiw4G49-AK8UHM8iDOrmeZnJEckO?usp=sharing). You can fork this repository and download the folder directly to the project.

**The file structure of the generated dataset**:

textual data|label|time
:-----:|:-----:|:-----:|
the text|"F":fraudulent, "NF": otherwise|"yyyymmdd" the issuing date of the report containing this text

**The result of running the respective files**

``` missing_file_to_df.py``` : process the *F* files not originally included in the shared dropbox by Allen. These files are collected manually, cleansed, and stored in ```HKEX Reports/Fraudulent/missing_files```

```non_fraudulent.py```: process the *NF* files included in the dropbox. They are stored in ```HKEX Reports/hkex_reports_annual``` and ```HKEX Reports/hkex_reports_semi-annual```. Note that all *F* files originally in the dropbox have been moved to ```HKEX Reports/Fraudulent``` already.

```given_file_to_df.py```: process the *F* files included in the dropbox. They are stored in ```HKEX Reports/Fraudulent```. 


