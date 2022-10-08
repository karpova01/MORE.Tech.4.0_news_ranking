def get_topics_LDA(data):
    import numpy as np
    import mglearn
    from nltk.stem.snowball import SnowballStemmer
    from sklearn.decomposition import LatentDirichletAllocation
    from nltk.corpus import stopwords
    from string import punctuation
    import re
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    nltk.download("stopwords")
    nltk.download('punkt')

    data = data.drop_duplicates(keep='first')

    data.loc[:,'News'] = data.loc[:,'Description'].apply(lambda x: ' '.join(re.findall('[а-яё]+', x.lower())))

# Создаем лемматизатор и набор стоп-слов
    russian_stopwords = stopwords.words("russian")

    def preprocess_text(text):
        tokens = [token for token in text if token not in russian_stopwords\
                  and token != " " \
                  and token not in punctuation]
        text = (tokens)
        return text

    word_token = []
    for word in data['News']:
        x = word.split()
        word_token.append(x)

    words = []
    for word in word_token:
        x = preprocess_text(word)
        words.append(x)

    stemmer = SnowballStemmer("russian")
    singles = []
    for w in words:
        single = [stemmer.stem(plural) for plural in w]
        singles.append(" ".join(single))

    vect = CountVectorizer(max_features=10000, max_df=.15)
    X = vect.fit_transform(singles)

    lda = LatentDirichletAllocation(learning_method="batch",
                                n_components= 10,
                                max_iter=25, random_state=0
                                )
    document_topics = lda.fit_transform(X)

    sorting = np.argsort(lda.components_, axis =1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names())
    top = mglearn.tools.print_topics(topics= range(10),
                           feature_names = feature_names,
                           sorting=sorting,
                           topics_per_chunk = 5, n_words = 10)
    
    return top

def get_insigt_image(data):
    import pandas as pd
    import regex as re
    import pymorphy2
    import random
    from matplotlib import rc
    import matplotlib.pyplot as plt
    from rutermextract import TermExtractor
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import IncrementalPCA
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
    import nltk
    import time
    nltk.download('stopwords')


    data = data.drop_duplicates(keep='first')
    
    data.loc[:,'News'] = data.loc[:,'Description'].apply(lambda x: ' '.join(re.findall('[а-яё]+', x.lower())))

    morph = pymorphy2.MorphAnalyzer()

    def lemmatize(txt):
        words = txt.split() 
        res = list()
        for word in words:
            p = morph.parse(word)[0]
            res.append(p.normal_form)
        return res

#data.loc[:,'News'] = data.loc[:,'News'].apply(lambda x: ' '.join(lemmatize(x)))

# извлечение ключевых слов
    term_extractor = TermExtractor()

    news_terms = []
    all_terms = []
    for news in data['News']:
        terms = []
        for term in term_extractor(news):
            terms.append(term.normalized)
            for i in range(term.count):
                all_terms.append(term.normalized)
        news_terms.append(terms)

# извлечение трендов по ключевым словам
    stopwords = nltk.corpus.stopwords.words('russian')
    stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на','россия',\
                  'год','сутки','страна','январь','февраль','март','апрель','май','июнь',\
                  'июль','август','сентябрь','октябрь','ноябрь','декабрь'])

    terms = [word for word in all_terms if not word in stopwords]

    key_words =  ' '.join(terms)
    trends = []
    for term in term_extractor(key_words):
        trends.append([term.normalized, term.count])

# Кластеризация новостей

    news_keyword = []
    for keywords in news_terms:
        news_keyword.append(' '.join(keywords))

    n_featur=1000
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.01, stop_words=stopwords,
                                 use_idf=True, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(news_keyword)

    num_clusters = 8

# Метод к-средних
    km = KMeans(n_clusters=num_clusters)
    #get_ipython().magic('time km.fit(tfidf_matrix)')
    idx = km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

# MiniBatchKMeans
    mbk  = MiniBatchKMeans(init='random', n_clusters=num_clusters) #(init='k-means++', ‘random’ or an ndarray)
    mbk.fit_transform(tfidf_matrix)
    mbk.fit(tfidf_matrix)
    miniclusters = mbk.labels_.tolist()

# DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(tfidf_matrix)
    labels = db.labels_
    labels.shape

# Аггломеративная класстеризация
    agglo1 = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean')
    answer = agglo1.fit_predict(tfidf_matrix.toarray())

#k-means
    clusterkm = km.labels_.tolist()
#minikmeans
    clustermbk = mbk.labels_.tolist()
    #dbscan
    clusters3 = labels


    frame = pd.DataFrame(news_keyword, index = [clusterkm])

#k-means
    out = { 'title': news_keyword, 'cluster': clusterkm }
    frame1 = pd.DataFrame(out, index = [clusterkm], columns = ['title', 'cluster'])

#mini
    out = { 'title': news_keyword, 'cluster': clustermbk }
    frame_minik = pd.DataFrame(out, index = [clustermbk], columns = ['title', 'cluster'])

#frame1['cluster'].value_counts()
#frame_minik['cluster'].value_counts()

    dist = 1 - cosine_similarity(tfidf_matrix)

# Метод главных компонент - PCA
    icpa = IncrementalPCA(n_components=2, batch_size=16)
    #get_ipython().magic('time icpa.fit(dist) #demo =')
    demo2 = icpa.transform(dist)
    xs, ys = demo2[:, 0], demo2[:, 1]

# PCA 3D
    icpa = IncrementalPCA(n_components=3, batch_size=16)
    #get_ipython().magic('time icpa.fit(dist) #demo =')
    ddd = icpa.transform(dist)
    xs, ys, zs = ddd[:, 0], ddd[:, 1], ddd[:, 2]


# цвета кластеров
    cluster_colors = {0: '#FF0000', 1: '#FF4500', 2: '#8B0000',  3: '#191970', 4: '#006400', 5: '#000000', 6: '#800080', 7: '#ff00ff',}
#имена кластерам
    cluster_names = {0: '0',  1: '1', 2: '2',  3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}
#matplotlib inline

#создаем data frame, который содержит координаты (из PCA) + номера кластеров и сами запросы
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusterkm, title=news_keyword)) 
#группируем по кластерам
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(72, 36))

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(        axis= 'x',          
                       which='both',      
                       bottom='off',      
                       top='off',         
                       labelbottom='off')
        ax.tick_params(        axis= 'y',         
                       which='both',     
                       left='off',      
                       top='off',       
                       labelleft='off')
    
    ax.legend(numpoints=1)  #показать легенду только 1 точки

    plt.savefig('insigt_image.png')
    plt.close()