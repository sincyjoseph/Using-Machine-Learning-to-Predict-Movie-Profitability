```python
import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import reduce
import winsound
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection  import GridSearchCV, permutation_test_score, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score 
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
```


```python
imdb = pd.read_csv("IMDb movies.csv")
```

    c:\users\harsh\desktop\env\lib\site-packages\IPython\core\interactiveshell.py:3062: DtypeWarning: Columns (3) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    


```python
imdb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>actors</th>
      <th>description</th>
      <th>avg_vote</th>
      <th>votes</th>
      <th>budget</th>
      <th>usa_gross_income</th>
      <th>worlwide_gross_income</th>
      <th>metascore</th>
      <th>reviews_from_users</th>
      <th>reviews_from_critics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0000009</td>
      <td>Miss Jerry</td>
      <td>Miss Jerry</td>
      <td>1894</td>
      <td>1894-10-09</td>
      <td>Romance</td>
      <td>45</td>
      <td>USA</td>
      <td>None</td>
      <td>Alexander Black</td>
      <td>...</td>
      <td>Blanche Bayliss, William Courtenay, Chauncey D...</td>
      <td>The adventures of a female reporter in the 1890s.</td>
      <td>5.9</td>
      <td>154</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0000574</td>
      <td>The Story of the Kelly Gang</td>
      <td>The Story of the Kelly Gang</td>
      <td>1906</td>
      <td>1906-12-26</td>
      <td>Biography, Crime, Drama</td>
      <td>70</td>
      <td>Australia</td>
      <td>None</td>
      <td>Charles Tait</td>
      <td>...</td>
      <td>Elizabeth Tait, John Tait, Norman Campbell, Be...</td>
      <td>True story of notorious Australian outlaw Ned ...</td>
      <td>6.1</td>
      <td>589</td>
      <td>$ 2250</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0001892</td>
      <td>Den sorte drøm</td>
      <td>Den sorte drøm</td>
      <td>1911</td>
      <td>1911-08-19</td>
      <td>Drama</td>
      <td>53</td>
      <td>Germany, Denmark</td>
      <td>NaN</td>
      <td>Urban Gad</td>
      <td>...</td>
      <td>Asta Nielsen, Valdemar Psilander, Gunnar Helse...</td>
      <td>Two men of high rank are both wooing the beaut...</td>
      <td>5.8</td>
      <td>188</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0002101</td>
      <td>Cleopatra</td>
      <td>Cleopatra</td>
      <td>1912</td>
      <td>1912-11-13</td>
      <td>Drama, History</td>
      <td>100</td>
      <td>USA</td>
      <td>English</td>
      <td>Charles L. Gaskill</td>
      <td>...</td>
      <td>Helen Gardner, Pearl Sindelar, Miss Fielding, ...</td>
      <td>The fabled queen of Egypt's affair with Roman ...</td>
      <td>5.2</td>
      <td>446</td>
      <td>$ 45000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0002130</td>
      <td>L'Inferno</td>
      <td>L'Inferno</td>
      <td>1911</td>
      <td>1911-03-06</td>
      <td>Adventure, Drama, Fantasy</td>
      <td>68</td>
      <td>Italy</td>
      <td>Italian</td>
      <td>Francesco Bertolini, Adolfo Padovan</td>
      <td>...</td>
      <td>Salvatore Papa, Arturo Pirovano, Giuseppe de L...</td>
      <td>Loosely adapted from Dante's Divine Comedy and...</td>
      <td>7.0</td>
      <td>2237</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
len(imdb)
```




    85855




```python
#remove values with no gross income data or budget data
imdb=imdb[imdb['worlwide_gross_income'].isna()==False]
imdb=imdb[imdb['budget'].isna()==False]
imdb.reset_index(drop=True, inplace=True)
```


```python
#Remove dollar sign from revenue and budget
worldwide=[]
budget=[]
for index, row in imdb.iterrows():
    try:
        worldwide.append(row['worlwide_gross_income'].split('$ ')[1])
    except:
        worldwide.append(0)
        
    budget.append(row['budget'].split(' ')[1])

imdb['worlwide_gross_income']=[int(x) for x in worldwide]
imdb['budget']=[int(x) for x in budget]
```


```python
len(imdb)
```




    12762




```python
#split multiple items in column into a list of separate items
imdb['country'] = (imdb['country'].str.split(', '))
imdb['genre'] = (imdb['genre'].str.split(', '))
imdb['language'] = (imdb['language'].str.split(', '))
imdb['writer'] = (imdb['writer'].str.split(', '))
imdb['director'] = (imdb['director'].str.split(', '))
```


```python
# perform sentiment analysis on description and title
des_scores=[]
title_scores=[]
analyzer = SentimentIntensityAnalyzer()
for x in imdb['description']:
    try:
        des_scores.append(analyzer.polarity_scores(x)['compound'])
    except TypeError:
        des_scores.append(0)

for x in imdb['title']:
    try:
        title_scores.append(analyzer.polarity_scores(x)['compound'])
    except TypeError:
        title_scores.append(0)
imdb['description_score']=des_scores
imdb['title']=title_scores
imdb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>description</th>
      <th>avg_vote</th>
      <th>votes</th>
      <th>budget</th>
      <th>usa_gross_income</th>
      <th>worlwide_gross_income</th>
      <th>metascore</th>
      <th>reviews_from_users</th>
      <th>reviews_from_critics</th>
      <th>description_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0010323</td>
      <td>0.0</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>1920</td>
      <td>1920-02-27</td>
      <td>[Fantasy, Horror, Mystery]</td>
      <td>76</td>
      <td>[Germany]</td>
      <td>[German]</td>
      <td>[Robert Wiene]</td>
      <td>...</td>
      <td>Hypnotist Dr. Caligari uses a somnambulist, Ce...</td>
      <td>8.1</td>
      <td>55601</td>
      <td>18000</td>
      <td>$ 8811</td>
      <td>8811</td>
      <td>NaN</td>
      <td>237.0</td>
      <td>160.0</td>
      <td>-0.4215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0011440</td>
      <td>0.0</td>
      <td>Markens grøde</td>
      <td>1921</td>
      <td>1921-12-02</td>
      <td>[Drama]</td>
      <td>107</td>
      <td>[Norway]</td>
      <td>NaN</td>
      <td>[Gunnar Sommerfeldt]</td>
      <td>...</td>
      <td>After the Nobel prize winning Knut Hamsun-nove...</td>
      <td>6.6</td>
      <td>195</td>
      <td>250000</td>
      <td>NaN</td>
      <td>4272</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.1779</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0012190</td>
      <td>0.0</td>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>1921</td>
      <td>1923-04-16</td>
      <td>[Drama, Romance, War]</td>
      <td>150</td>
      <td>[USA]</td>
      <td>[None]</td>
      <td>[Rex Ingram]</td>
      <td>...</td>
      <td>An extended family split up in France and Germ...</td>
      <td>7.2</td>
      <td>3058</td>
      <td>800000</td>
      <td>$ 9183673</td>
      <td>9183673</td>
      <td>NaN</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>-0.7579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0012349</td>
      <td>0.0</td>
      <td>The Kid</td>
      <td>1921</td>
      <td>1923-11-26</td>
      <td>[Comedy, Drama, Family]</td>
      <td>68</td>
      <td>[USA]</td>
      <td>[English, None]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>The Tramp cares for an abandoned child, but ev...</td>
      <td>8.3</td>
      <td>109038</td>
      <td>250000</td>
      <td>NaN</td>
      <td>26916</td>
      <td>NaN</td>
      <td>173.0</td>
      <td>105.0</td>
      <td>-0.6310</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0014624</td>
      <td>0.0</td>
      <td>A Woman of Paris: A Drama of Fate</td>
      <td>1923</td>
      <td>1927-06-06</td>
      <td>[Drama, Romance]</td>
      <td>82</td>
      <td>[USA]</td>
      <td>[None, English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>A kept woman runs into her former fiancé and f...</td>
      <td>7.0</td>
      <td>4735</td>
      <td>351000</td>
      <td>NaN</td>
      <td>11233</td>
      <td>NaN</td>
      <td>37.0</td>
      <td>24.0</td>
      <td>0.6908</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
#get dummy variables for categories that need them
df = pd.get_dummies(imdb['genre'].apply(pd.Series).stack()).sum(level=0)
df2 = pd.get_dummies(imdb['language'].apply(pd.Series).stack()).sum(level=0)
df3 = pd.get_dummies(imdb['director'].apply(pd.Series).stack()).sum(level=0)
df4 = pd.get_dummies(imdb['writer'].apply(pd.Series).stack()).sum(level=0)
df5 = pd.get_dummies(imdb['country'].apply(pd.Series).stack()).sum(level=0)
```


```python
df = df.add_prefix('genre_')
df2 = df2.add_prefix('langauge_')
df3 = df3.add_prefix('director_')
df4 = df4.add_prefix('writer_')
df5 = df5.add_prefix('country_')
```


```python
#add similar column for merging
df['imdb_title_id']=imdb['imdb_title_id']
df2['imdb_title_id']=imdb['imdb_title_id']
df3['imdb_title_id']=imdb['imdb_title_id']
df4['imdb_title_id']=imdb['imdb_title_id']
df5['imdb_title_id']=imdb['imdb_title_id']
```


```python
#merge all dataframes into one
dfs = [imdb,df,df2,df3,df4,df5]
df_final = reduce(lambda left,right: pd.merge(left,right,on='imdb_title_id'), dfs)
```


```python
df_final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>country_UK</th>
      <th>country_USA</th>
      <th>country_Ukraine</th>
      <th>country_United Arab Emirates</th>
      <th>country_Uruguay</th>
      <th>country_Venezuela</th>
      <th>country_Vietnam</th>
      <th>country_West Germany</th>
      <th>country_Yemen</th>
      <th>country_Yugoslavia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0010323</td>
      <td>0.0</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>1920</td>
      <td>1920-02-27</td>
      <td>[Fantasy, Horror, Mystery]</td>
      <td>76</td>
      <td>[Germany]</td>
      <td>[German]</td>
      <td>[Robert Wiene]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0012190</td>
      <td>0.0</td>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>1921</td>
      <td>1923-04-16</td>
      <td>[Drama, Romance, War]</td>
      <td>150</td>
      <td>[USA]</td>
      <td>[None]</td>
      <td>[Rex Ingram]</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0012349</td>
      <td>0.0</td>
      <td>The Kid</td>
      <td>1921</td>
      <td>1923-11-26</td>
      <td>[Comedy, Drama, Family]</td>
      <td>68</td>
      <td>[USA]</td>
      <td>[English, None]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0014624</td>
      <td>0.0</td>
      <td>A Woman of Paris: A Drama of Fate</td>
      <td>1923</td>
      <td>1927-06-06</td>
      <td>[Drama, Romance]</td>
      <td>82</td>
      <td>[USA]</td>
      <td>[None, English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0015864</td>
      <td>0.0</td>
      <td>The Gold Rush</td>
      <td>1925</td>
      <td>1925-10-23</td>
      <td>[Adventure, Comedy, Drama]</td>
      <td>95</td>
      <td>[USA]</td>
      <td>[English, None]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12654</th>
      <td>tt9878242</td>
      <td>0.0</td>
      <td>Subharathri</td>
      <td>2019</td>
      <td>2019-07-06</td>
      <td>[Drama, Romance]</td>
      <td>130</td>
      <td>[India]</td>
      <td>[Malayalam]</td>
      <td>[Vyasan K.P.]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12655</th>
      <td>tt9886872</td>
      <td>0.0</td>
      <td>Munthiri Monchan</td>
      <td>2019</td>
      <td>2019-12-06</td>
      <td>[Comedy, Romance]</td>
      <td>130</td>
      <td>[India]</td>
      <td>[Malayalam]</td>
      <td>[Vijith Nambiar]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12656</th>
      <td>tt9894394</td>
      <td>0.0</td>
      <td>Upin &amp; Ipin: Keris Siamang Tunggal</td>
      <td>2019</td>
      <td>2019-03-21</td>
      <td>[Animation]</td>
      <td>100</td>
      <td>[Malaysia]</td>
      <td>[Malay]</td>
      <td>[Adam Bin Amiruddin, Syed Nurfaiz Khalid bin S...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12657</th>
      <td>tt9900782</td>
      <td>0.0</td>
      <td>Kaithi</td>
      <td>2019</td>
      <td>2019-10-25</td>
      <td>[Action, Thriller]</td>
      <td>145</td>
      <td>[India]</td>
      <td>[Tamil]</td>
      <td>[Lokesh Kanagaraj]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12658</th>
      <td>tt9905412</td>
      <td>0.0</td>
      <td>Ottam</td>
      <td>2019</td>
      <td>2019-03-08</td>
      <td>[Drama]</td>
      <td>120</td>
      <td>[India]</td>
      <td>[Malayalam]</td>
      <td>[Zam]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12659 rows × 20275 columns</p>
</div>




```python
# get a binary column to represent whether a movie was able to recoup its production costs
profitable = []
for index, row in df_final.iterrows():
    if row['worlwide_gross_income']>row['budget']:
        profitable.append(1)
    else:
        profitable.append(0)
df_final['profitable']=profitable
```


```python
df_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>country_USA</th>
      <th>country_Ukraine</th>
      <th>country_United Arab Emirates</th>
      <th>country_Uruguay</th>
      <th>country_Venezuela</th>
      <th>country_Vietnam</th>
      <th>country_West Germany</th>
      <th>country_Yemen</th>
      <th>country_Yugoslavia</th>
      <th>profitable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0010323</td>
      <td>0.0</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>1920</td>
      <td>1920-02-27</td>
      <td>[Fantasy, Horror, Mystery]</td>
      <td>76</td>
      <td>[Germany]</td>
      <td>[German]</td>
      <td>[Robert Wiene]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0012190</td>
      <td>0.0</td>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>1921</td>
      <td>1923-04-16</td>
      <td>[Drama, Romance, War]</td>
      <td>150</td>
      <td>[USA]</td>
      <td>[None]</td>
      <td>[Rex Ingram]</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0012349</td>
      <td>0.0</td>
      <td>The Kid</td>
      <td>1921</td>
      <td>1923-11-26</td>
      <td>[Comedy, Drama, Family]</td>
      <td>68</td>
      <td>[USA]</td>
      <td>[English, None]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0014624</td>
      <td>0.0</td>
      <td>A Woman of Paris: A Drama of Fate</td>
      <td>1923</td>
      <td>1927-06-06</td>
      <td>[Drama, Romance]</td>
      <td>82</td>
      <td>[USA]</td>
      <td>[None, English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0015864</td>
      <td>0.0</td>
      <td>The Gold Rush</td>
      <td>1925</td>
      <td>1925-10-23</td>
      <td>[Adventure, Comedy, Drama]</td>
      <td>95</td>
      <td>[USA]</td>
      <td>[English, None]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 20276 columns</p>
</div>




```python
#get the feature and label sets
#remove all the features that wouldn't be known until after a movie is produced
X = df_final.drop(['imdb_title_id','title','original_title','date_published',
                   'genre', 'duration', 'country', 'language', 'director', 'writer',
                   'production_company', 'actors', 'description', 'avg_vote', 'votes',
                   'usa_gross_income', 'worlwide_gross_income', 'metascore','reviews_from_users', 
                   'reviews_from_critics', 'description_score'],axis=1)
y = df_final['profitable']
```


```python
corlist=pd.DataFrame(X.corrwith(y),columns=['coor'])
```


```python
corlist.reset_index(inplace=True)
```


```python
df1 = corlist[corlist['coor']>.025]
df2 = corlist[corlist['coor']<-.025]
df = pd.concat([df1,df2])
```


```python
print(len(X.columns))
print(len(df))
```

    20255
    200
    


```python
X = X[df['index']]
X.drop('profitable',axis=1,inplace=True)
```


```python
X.dropna(inplace=True)
```


```python
# generate more samples from the data to balance out binary outcomes
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
```


```python
#get a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=52)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=13)
```


```python
# standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)
```


```python
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train,y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
# predict values using X test and the new model
y_pred = classifier.predict(X_val)
```


```python
#print the confusion matrix as well as multiple scores to evaluate the model
cm = confusion_matrix(y_val, y_pred)
print(cm)
print('Accuracy: ' + str(accuracy_score(y_val, y_pred)))
print('AUC: ' + str(roc_auc_score(y_val, y_pred)))
print('F1 Score: ' + str(f1_score(y_val, y_pred)))
```

    [[985 467]
     [507 951]]
    Accuracy: 0.6652920962199312
    AUC: 0.6653190150664897
    F1 Score: 0.6613351877607787
    


```python
#use grid search cv to run multiple random forest classifier models to find best hyperparameters
param_grid = { 
    'max_depth':[3,5,7,11,None],
    'n_estimators': [50, 100, 250,500],
    'max_features': ['auto','sqrt','log2'],
}

CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 5,n_jobs=-1)
CV_rfc.fit(X_val, y_val)
print(CV_rfc.best_estimator_)
```

    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=11, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)
    


```python
winsound.Beep(440,250)
```


```python
# predict values using X test and the new model
y_pred = CV_rfc.best_estimator_.predict(X_test)
```


```python
#print the confusion matrix as well as multiple scores to evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('AUC: ' + str(roc_auc_score(y_test, y_pred)))
print('F1 Score: ' + str(f1_score(y_test, y_pred)))
```

    [[ 991  446]
     [ 463 1010]]
    Accuracy: 0.6876288659793814
    AUC: 0.6876533341270212
    F1 Score: 0.689655172413793
    


```python
# perform multiple random permutations to find p-value 
clf = CV_rfc.best_estimator_
cv = StratifiedKFold(2, shuffle=True, random_state=35)

score_orig, perm_scores_orig, pvalue_orig = permutation_test_score(
    clf, X_test, y_test, scoring="accuracy", cv=cv, n_permutations=1000)
```


```python
pvalue_orig
```




    0.000999000999000999




```python
winsound.Beep(440,250)
```


```python
logisticRegr = LogisticRegression(random_state=21,n_jobs=-1)
logisticRegr.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=-1, penalty='l2', random_state=21,
                       solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)




```python
y_pred = logisticRegr.predict(X_val)
```


```python
#print the confusion matrix as well as multiple scores to evaluate the model
cm = confusion_matrix(y_val, y_pred)
print(cm)
print('Accuracy: ' + str(accuracy_score(y_val, y_pred)))
print('AUC: ' + str(roc_auc_score(y_val, y_pred)))
print('F1 Score: ' + str(f1_score(y_val, y_pred)))
```

    [[ 884  568]
     [ 411 1047]]
    Accuracy: 0.663573883161512
    AUC: 0.6634612114410094
    F1 Score: 0.6814188089814512
    


```python
#use grid search cv to run multiple random forest classifier models to find best hyperparameters
param_grid = {
    'max_iter':[50,100,200,None],
    'solver':['newton-cg', 'lbfgs', 'liblinear','sag','saga'],
    'penalty':['none','l1','l2','elasticnet'],
    'C':[100, 10, 1.0, 0.1, 0.01]
}

CV_logreg = GridSearchCV(estimator=logisticRegr, param_grid=param_grid, cv= 5,n_jobs=-1)
CV_logreg.fit(X_val, y_val)
print(CV_logreg.best_estimator_)
```

    LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=50,
                       multi_class='auto', n_jobs=-1, penalty='l1', random_state=21,
                       solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
    

    c:\users\harsh\desktop\env\lib\site-packages\sklearn\linear_model\_logistic.py:1537: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 4.
      warnings.warn("'n_jobs' > 1 does not have any effect when"
    


```python
y_pred = CV_logreg.predict(X_test)
```


```python
#print the confusion matrix as well as multiple scores to evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('AUC: ' + str(roc_auc_score(y_test, y_pred)))
print('F1 Score: ' + str(f1_score(y_test, y_pred)))
```

    [[ 795  642]
     [ 383 1090]]
    Accuracy: 0.647766323024055
    AUC: 0.646611165204722
    F1 Score: 0.6801872074882994
    


```python
winsound.Beep(440,250)
```


```python
# perform multiple random permutations to find p-value 
clf = CV_logreg.best_estimator_
cv = StratifiedKFold(2, shuffle=True, random_state=35)

score_orig, perm_scores_orig, pvalue_orig = permutation_test_score(
    clf, X_test, y_test, scoring="accuracy", cv=cv, n_permutations=1000,n_jobs=None)
```



```python
pvalue_orig
```




    0.000999000999000999




```python

```
