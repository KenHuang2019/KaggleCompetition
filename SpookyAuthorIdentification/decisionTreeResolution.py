import os
import sys
import json

import nltk
import jieba
import string

import graphviz  # Visualization Tool
import argparse
import numpy as np
import pandas as pd

from time import time
from datetime import datetime
from nltk.corpus import stopwords

from sklearn import tree
from sklearn.preprocessing import LabelEncoder  # Numerical Conversion Tool
from sklearn.model_selection import cross_val_score, GridSearchCV , train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# 若使用 GridSearchCV，可給定一個 set 的 parameters ，GridSearchCV會嘗試所有排列組合的可能
param_grid = {
    "criterion": ["gini", "entropy"],
    "min_samples_split": [1.0, 3, 5, 7, 10],
    "max_depth": [6, 7, 8, 10],
    "min_samples_leaf": [5, 6, 7, 8, 9, 10],
    "max_leaf_nodes": [None, 10],
    "min_impurity_decrease": [0.0,  0.1, 0.3],
    "ccp_alpha":[0.0, 0.05, 0.1, 0.3],
    "random_state":[None, 1, 10],
    "splitter": ["best", "random"],
    "max_features": [None, "auto"]
}

def load_data(path):
    """
    載入資料集
    查看基本資訊
    """
    data = pd.read_csv(path, sep=',')# load data
    if data.shape[0] > 0:
        print(data.shape)  # 確認資料筆數、column 數量
        print(data.head()) # 查看前五筆資料
        print(data.isnull().sum()) # 確認column 名稱、是否有空值
        return data
    else:
        print('There is no data!')
        return None

def tokenize(text, stops, puns):
    if pd.notnull(text):
        tokens = nltk.wordpunct_tokenize(text) # 使用nltk.wordpunct_tokenize將text切開
        words = [t.lower() for t in tokens if t not in stops and t not in puns]  # 去掉停用字與標點符號，並轉小寫

        return words
    else:
        return None

def load_gloves():
    """
    使用GloVe(Global Vectors for Word Representation) 做word to vector
    https://nlp.stanford.edu/projects/glove/
    """
    w_v_pairs = {} # word vector pairs
    glove_path = r'.\glove\glove.6B.50d.txt'# 這文檔的每一行是「word」並配上其「vector」(空白鍵分隔)

    # 將每一行開頭的 word 作為 dictonary 的 key，後面的 vector 做為 value，並加入到 w_v_pairs
    with open(glove_path, 'r', encoding='utf-8') as f: 
        for l in f:  
            s_l = l.split() 
            word = s_l[0]
            vector = s_l[1:]
            w_v_pairs[word] = np.asarray(vector, dtype=np.float32) 

    # 若GloVe文檔使用glove.840B.300d，則適用於這種轉換方式
    # with open(glove_path, 'r', encoding='utf-8') as f:
    #     for l in f:
    #         values = l.split() # 逐行切開
    #         word = ''.join(values[:-300])
    #         w_v_pairs[word] = np.asarray(values[-300:], dtype='float32')

    v_dim =  len(w_v_pairs.get('men')) # 為避免後續對應不到vector，需要vector長度資訊補零
    print("vec_dimensions:",v_dim)
    print("w_v_pairs length:", len(list(w_v_pairs.items())))

    return w_v_pairs, v_dim

def w2v(tokenized_words, word_vector_pairs, _v_dim):
    """
    將 tokenized words 和 glove 做 mapping
    """
    # 透過 key value pair 做 mapping
    tmp_word_vectors = [word_vector_pairs.get(w) for w in tokenized_words if w in word_vector_pairs.keys()] 
    w_v = np.average(np.array(tmp_word_vectors), axis=0) # 將mapping到的所有vector取平均，作為word的單一vector

    # 若找不到對應的詞向量，則給一條全部為零的向量
    if np.sum(np.isnan(w_v)) > 0:
        w_v= np.zeros(v_dim, ) # 長度為原詞彙代表向量的長度

    # print("word vector: ", w_v)
    return w_v

def deconstruct_vector(f_c, i, w_v):
    """
    f_c : features container
    i : index
    w_v : word vector

    拆解 vector ，將每個數值作為一個特徵，讓decision tree可以分類
    """
    for v in range(len(w_v)): # v 是第幾個特徵的數字
        if i == 0:
            f_c['f' + str(v)] = [w_v[v]]
        else:
            f_c['f' + str(v)].append(w_v[v])

    return f_c


def tfidf_preprocess(item):
    # 把 text 用 jieba.cut 進行分詞
    terms = [t for t in jieba.cut(item, cut_all=True)]  # cut_all 設為 True，可比對的詞彙比較多
    return terms

def t2f(df, w_v_pairs, v_dim):
    """
    df = data
    text 欄位本身長度當作一種 feature
    word數量也當作一種 feature
    再把text欄位的句子轉換成 vector
    取平均、標準差、中位數作為 feature
    然後將vecotr中的每個數字拆解出來也做為 feature
    """
    # 載入英文停用字與標點符號
    stops = set(stopwords.words('english'))
    puns = string.punctuation

    # 建立容器裝vecotr拆解後的數字做為特徵
    features = {
        't_len':[],
        'w_len':[],
        'v_mean':[],
        'noun_count':[],
        'adj_count':[],
        'verbs_count':[],
        'noun_ratio':[],
        'adj_ratio':[],
        'verbs_ratio':[],
    }

    # 將 text 轉換為 TFIDF 的 vector
    df['tfidf_vector'] = list(
        TfidfVectorizer().fit_transform(
            df['text'].apply(tfidf_preprocess).apply( lambda x:" ".join(x) ).values).toarray() # 把每個詞彙用空白串起來之後做TFIDF計算
     )

    df['unique_tfidf_vector'] = df['tfidf_vector'].apply(lambda x:np.unique(x))
    df['unique_tfidf_vector_mean'] = df['unique_tfidf_vector'].apply(lambda x: np.mean(x))

    df['tfidf_v_mean'] = df['tfidf_vector'].apply(lambda x: np.mean(x))
    # tfidf vector normalization
    max_mean = df['tfidf_v_mean'].max(axis = 0)
    min_mean = df['tfidf_v_mean'].min(axis = 0)
    df['tfidf_v_normalized_mean'] = df['tfidf_v_mean'].apply( 
        lambda x: ( x - min_mean ) / ( max_mean - min_mean ) *10
    )

    # 將各種文字類型作為 feature
    df["unique_w_len"] = df["text"].apply(lambda x: len(set(str(x).split())))
    df["num_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))
    df["num_punctuations"] =df['text'].apply(lambda x: len([c for c in str(x) if c in puns]) )
    df["num_words_upper"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df["num_words_title"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["mean_word_len"] = df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # 將data逐行取出
    for row_data in df.itertuples():
        i = row_data[0] # index
        print("Index: ", i)
        text = row_data[2]

        features['t_len'].append(len(text)) # 文本長短當作一種 feature

        # 將text欄位的句子去除不必要的詞彙與標點符號，再拆解成單字放入list
        words = tokenize(text, stops, puns)
        features['w_len'].append(len(words))

        # 將詞性轉換成一種 feature
        pos_list = nltk.pos_tag(words)
        n_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
        adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
        v_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
        # 計算使用不同詞性的數量
        features['noun_count'].append(n_count) # 名詞
        features['adj_count'].append(adj_count) # 形容詞
        features['verbs_count'].append(v_count) # 動詞
        # 計算使用不同詞性的比率
        features['noun_ratio'].append(n_count / len(words)) # 名詞
        features['adj_ratio'].append(adj_count / len(words)) # 形容詞
        features['verbs_ratio'].append(v_count / len(words)) # 動詞

        # 將多個詞彙轉換成單一個 vector
        w_v = w2v(words, w_v_pairs, v_dim) # word vector 
        features['v_mean'].append(np.mean(w_v)) # 取平均作為其中一種feature

        # 將vector拆開當作不同feature
        features = deconstruct_vector(features, i, w_v)

    return features


def report(grid_scores, cv, n_top=3):
    """
    grid_scores 是 GridSearchCV 的 output
    n_top 是將排名前n的 parameters 輸出，預設值是3
    """

    top_n_data = {
        'mean_test_score':[],
        'std_test_score':[],
        'params':[]
    }
    counter = 0
    print("---------------------------------------------------------------------------------")

    print("GridSearchCV search resualts: ")
    for i in range(n_top):
        ranking_arr = np.where(grid_scores['rank_test_score'] == i+1)[0]
        arr_len = ranking_arr.shape[0]
        if arr_len > 0:
            for j in range(arr_len):
                index = ranking_arr[j]
                print("""Rank: {}
╠═ mean_test_score: {}
╠═ std_test_score: {}
╚═ params: {} """.format(
                        i+1,
                        grid_scores['mean_test_score'][index],
                        grid_scores['std_test_score'][index],
                        grid_scores['params'][index],
                    )
                )
                top_n_data['mean_test_score'].append(grid_scores['mean_test_score'][index])
                top_n_data['std_test_score'].append(grid_scores['std_test_score'][index])
                top_n_data['params'].append(grid_scores['params'][index])
                print("---------------------------------------------------------------------------------")
                counter += 1
                if counter == n_top:
                    break

    with open('./test_result/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'_cv_'+str(cv)+'_author_id_record.json', 'w') as outfile:
        json.dump(top_n_data, outfile)

    return None

def run_gridsearch(X, y, clf, param_grid, cv=5):
    """
    X 是前處理之後的特徵
    y 是要預測的目標
    clf 是 Decision Tree 模型
    param_grid 則是要嘗試的參數組
    cv cross-validation的參數
    """
    grid_search = GridSearchCV(clf,
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=-1)
    start = time()
    grid_search.fit(X, y)

    print("\nGridSearchCV took {:.2f} seconds for {:d} kinds of parameters setting.".format(time() - start, len(grid_search.cv_results_['rank_test_score'])))

    report(grid_search.cv_results_, cv, 20)
    return None

if  __name__ == "__main__":
    train_data_path = r'.\dataset\Spooky_Author_Identification\train.csv'

    preprocessed_train_data_path = r'.\dataset\Spooky_Author_Identification\preprocessed_train.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", required=False, default='n')
    parser.add_argument("-u", "--use_grid_search", required=False, default='n')
    args = parser.parse_args()

    if args.preprocess == 'y' or  args.preprocess == 'Y':
        train_data = load_data(train_data_path)

        # 載入 GloVe (Global Vectors for Word Representation)
        word_vector_pairs, v_dim = load_gloves()

        # 將 text 欄位的句子轉換成 feature
        train_features = t2f(train_data, word_vector_pairs, v_dim)

        # 將作者名稱轉換為數字標籤 ，EAP = 0，HPL = 1，MWS = 2
        train_data['numerical_author_label'] =  LabelEncoder().fit_transform(train_data['author'])

        # 將每個 feature mapping 成 data frame
        for k in train_features.keys():
            train_data[k] = pd.Series(train_features[k], dtype='float64')

        # 將前處理過後的資料儲存起來，後續調參就可不用重新處理
        train_data.to_csv (preprocessed_train_data_path, index = False, header=True) 
    else:
        train_data = load_data(preprocessed_train_data_path)

    y = train_data["numerical_author_label"]
    x = train_data.drop(["id", "text", "author", "numerical_author_label", "tfidf_vector", "unique_tfidf_vector", "tfidf_v_mean", "unique_tfidf_vector_mean", "tfidf_v_normalized_mean"], axis=1)

    x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size = 0.25)
    print(x_train.columns)

    if args.use_grid_search == 'y' or  args.use_grid_search == 'Y':
        # find parameters
        t = tree.DecisionTreeClassifier()
        for num_cv in range(20, 21):
            print('*** cv: ', num_cv)
            run_gridsearch(x, y, t, param_grid, cv=num_cv)
    else:
        # build model
        t = tree.DecisionTreeClassifier(
            criterion='gini',
            max_depth=7,
            max_features=None,
            max_leaf_nodes=None,
            min_samples_split=3,
            random_state=None,
            ccp_alpha=0.0,
            min_impurity_decrease=0.0,
            min_samples_leaf=10,
            splitter="best",
        )

        # train model
        t = t.fit(x_train, y_train)

        # prediction
        y_pred = t.predict(x_test)#Accuracy
        print("Accuracy score: ", accuracy_score(y_test, y_pred))

        # Visualize the tree model
        dot_data = tree.export_graphviz(
            t,
            out_file='graphviz.dot',
            label='all',
            impurity=True,
            proportion=True,
            feature_names=list(x),
            class_names=['EAP', 'HPL', 'MWS'],
            filled=True,
            rounded=True
        )

        # save tree image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.system("dot -T png graphviz.dot -o ./output/" + str(timestamp) + "_SAI.png")
