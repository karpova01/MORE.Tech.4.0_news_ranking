from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from insight import get_topics_LDA
from insight import get_insigt_image
app = FastAPI()

@app.get("/get-digest-boss")
def get_digest_boss():

    return get_digest('генеральный директор ресторана')

@app.get("/get-digest-buchhalter")
def get_digest_buchhalter():

    return get_digest('бухгалтер')

@app.get("/get-insight")
def get_insight():
    import pathlib
    from pathlib import Path
    import pandas as pd
    dir_path = pathlib.Path.cwd()
    path = Path(dir_path, 'export_dataframe.csv')
    data = pd.read_csv(path)
    return get_topics_LDA(data)
@app.get("/get-insight-image") 
def get_insight_image():
    import pathlib
    from pathlib import Path
    import pandas as pd
    dir_path = pathlib.Path.cwd()
    path = Path(dir_path, 'data_business.csv')
    data = pd.read_csv(path)
    get_insigt_image(data)
    file = open('insigt_image.png', mode="rb")

    return StreamingResponse(file, media_type="image/png")

def get_digest(role):
    import pandas as pd
    import regex as re
    import pymorphy2
    import nltk
    from rutermextract import TermExtractor
    from fuzzywuzzy import fuzz
    import pathlib
    from pathlib import Path
    
    dir_path = pathlib.Path.cwd()
    if role == 'бухгалтер':
        path = Path(dir_path, 'content','accounting_data.csv')
        data = pd.read_csv(path)
    elif 'генеральный директор' in role:
        path = Path(dir_path, 'content','data_business.csv')
        data = pd.read_csv(path)

    date = []
    for public_date in data['Publication Date']:
        date.append(public_date[:16])

    data['Date'] = date
    del data['Publication Date']

    data.loc[:,'News'] = data.loc[:,'Description'].apply(lambda x: ' '.join(re.findall('[а-яё]+', x.lower())))

    morph = pymorphy2.MorphAnalyzer()

    def lemmatize(txt):
        words = txt.split() 
        res = list()
        for word in words:
            p = morph.parse(word)[0]
            res.append(p.normal_form)
        return res

    data.loc[:,'News'] = data.loc[:,'News'].apply(lambda x: ' '.join(lemmatize(x)))

    digest = pd.DataFrame(columns=list(data.columns))

    if role == 'бухгалтер':
        for news in data['News']:
            if ('подписать' in news and 'закон' in news) or ('работодатель' in news and 'должный'in news) or\
            ('аккредитация' in news) or ('обязывать' in news  and 'закон' in news) or ('законопроект' in news and 'госдума' in news)\
            or ('сотрудник' in news and 'работодатель' in news) or 'отчётность' in news:
                digest = digest.append(data[data['News']==news],ignore_index=True)

    elif role == 'генеральный директор ресторана':
        for news in data['News']:
            if 'еда' in news or 'сыр' in news or 'семя' in news or 'зерно'in news or\
            'агропромышленный' in news or 'напиток' in news  or 'вкус' in news or 'бутылка' in news or\
            'маркет' in news or 'кола' in news or 'ингредиент' in news or 'сельскохозяйственный' in news:
                digest = digest.append(data[data['News']==news],ignore_index=True)

    elif role == 'генеральный директор автосалона':
        for news in data['News']:
            if 'автоконцерн' in news or 'автоваз' in news or 'автомобиль' in news or 'деталь'in news or\
            'комплектующие' in news or 'дизтопливо' in news  or 'топливо' in news\
            or 'автокредитование' in news or 'шина' in news:
                digest = digest.append(data[data['News']==news],ignore_index=True)

    indx = 0
    while indx < len(digest)-1:
        title_1 = digest.iloc[indx,0]
        j = indx+1
        while j < len(digest):
            title_2 = digest.iloc[j,0]
            if fuzz.partial_ratio(title_1,title_2)>50:
                digest.drop(digest.index[j], inplace=True)
            j+=1
        indx+=1
    answer = []


    if len(digest)<2:
        answer.append(f'Новость 1:{digest.iloc[0,0]}\n{digest.iloc[0,2]}')
        answer.append(f'Новость 2:{digest.iloc[1,0]}\n{digest.iloc[1,2]}')
    else:
        answer.append(f'Новость 1:{digest.iloc[0,0]}\n{digest.iloc[0,2]}')
        answer.append(f'Новость 2:{digest.iloc[1,0]}\n{digest.iloc[1,2]}')
        answer.append(f'Новость 3:{digest.iloc[2,0]}\n{digest.iloc[2,2]}')
        
    return answer