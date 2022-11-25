import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Project./All.csv',encoding='utf-8')  # DO CHANGE this directory

# transform label to {0,1}
alphabet_to_num={"NF":0,"F":1,}
df["label"]=df["label"].map(lambda x:alphabet_to_num[x])

#sort by timestamp
df=df.sort_values("time",ascending=False)
df=df.dropna()

#train-test split
df_train, df_test, = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# fit model
model = LogisticRegression()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(TfidfVectorizer(min_df=5, encoding='utf-8',lowercase=False,stop_words='english',max_features=10000, ngram_range=(1, 2)).fit_transform(df["textual data"]).toarray(), df['label'], df.index,stratify=df['label'], test_size=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# draw diagram for result
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=list(alphabet_to_num.keys()), yticklabels=list(alphabet_to_num.keys()))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
