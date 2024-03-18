#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk 
import re
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import pickle
import seaborn as sns


# In[3]:


import nltk
nltk.download('stopwords')


# In[4]:


df = pd.read_csv("IMDB Dataset.csv")


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df['sentiment'].unique()


# In[11]:


df['sentiment'].value_counts()


# In[12]:


label = LabelEncoder()
df['sentiment']= label.fit_transform(df['sentiment'])


# In[13]:


df.head()


# In[14]:


x= df['review']
y= df['sentiment']


# In[15]:


ps = PorterStemmer ()
corpus = []

for i in range(len(x)):
    print (i)
    review = re.sub("[^a-zA-Z]", " ", x[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words ("english"))]
    review =" ".join(review)
    corpus.append(review)


# In[18]:


corpus


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
cv= TfidfVectorizer(max_features= 5000)
x= cv.fit_transform(corpus).toarray()
Y= cv.fit_transform(corpus).toarray()


# In[20]:


x.shape,Y.shape


# In[21]:


x_train, x_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state = 101)


# In[22]:


x_train.shape, x_test.shape, Y_train.shape, Y_test.shape


# In[23]:


mnb= MultinomialNB()


# In[24]:


mnb.fit(x_train, Y_train)


# In[25]:


pred = mnb.predict(x_test)


# In[26]:


print(accuracy_score(Y_test , pred))
print(confusion_matrix(Y_test , pred))
print(classification_report(Y_test , pred))


# In[27]:


pd.DataFrame(np.c_[Y_test, pred] , columns=["Actual" , "Predicted"])


# In[28]:


pickle.dump(cv, open("count-vectorizer.pkl", "wb"))
pickle.dump(mnb, open("Movies_Review_Classification.pkl", "wb"))


# In[29]:


save_cv = pickle.load(open("count-vectorizer.pkl", "rb"))
model = pickle.load(open("Movies_Review_Classification.pkl", "rb"))


# In[30]:


def test_model(sentence):
    sen = save_cv.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 1:
        return 'positive review'
    else:
        return 'negative review'


# In[40]:


sen = 'The worst movie ever'
res = test_model(sen)
print(res)


# In[42]:


sen = 'The movie is good'
res = test_model(sen)
print(res)


# In[43]:


sen = 'The movie is not nice'
res = test_model(sen)
print(res)


# In[50]:


sen = 'This is the best movie of my life'
res = test_model(sen)
print(res)


# In[ ]:




