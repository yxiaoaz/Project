import os

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocessing


if __name__ == '__main__':

    if os.path.exists('Stemmed_All.csv'):
        df = pd.read_csv('Stemmed_All.csv', encoding='utf-8')  # ./Project./All.csv
    else:
        df=preprocessing("All.csv").run()


    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(TfidfVectorizer(min_df=5, encoding='utf-8',lowercase=False,max_features=10000, ngram_range=(1, 2)).fit_transform(df["textual data"]).toarray(), df['label'], df.index,stratify=df['label'], test_size=0.2, random_state=0)
    random_state=42
    smt = SMOTEENN(random_state=random_state)
    X_train, y_train = smt.fit_resample(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring_metric='balanced_accuracy'

    rf = RandomForestClassifier(random_state=random_state)
    param_grid = {
        'n_estimators': [100],
        'criterion': ['entropy', 'gini'],
        'bootstrap': [True, False],
        'max_depth': [6],
        'min_samples_leaf': [2, 3, 5],
        'min_samples_split': [2, 3, 5]
    }

    rf_clf = GridSearchCV(estimator=rf,
                          param_grid=param_grid,
                          scoring=scoring_metric,
                          cv=cv,
                          verbose=False,
                          n_jobs=1)

    y_pred_acc = rf_clf.fit(X_train, y_train).predict(X_test)
    # New Model Evaluation metrics
    print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
    print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
    print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
    print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

    #Logistic Regression (Grid Search) Confusion matrix
    confusion_matrix(y_test,y_pred_acc)

