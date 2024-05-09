import joblib
import pandas as pd
import numpy as np
import datetime as dt
import re
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2500)
pd.set_option('display.expand_frame_repr', False)

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

metadata = pd.read_json('games_metadata.json', lines=True)
metadata.to_parquet("tag_data.parquet")
metadata['tags'] = metadata['tags'].apply(lambda x: ', '.join(x))
metadata['tags'] = metadata['tags'].apply(lambda x: np.nan if x == '' else x)
metadata['tags'].isnull().sum()
metadata.dropna(inplace=True)



games = pd.read_csv('games.csv')
nodlcgames = games['title'].str.contains('DLC', case=False)
games = games[~nodlcgames]
noSoundtrackgames = games['title'].str.contains('Soundtrack', case=False)
games = games[~noSoundtrackgames]

games['date_release'] = pd.to_datetime(games['date_release'])
twenty_years_ago = datetime.now() - timedelta(days=365*15)
filtered_games = games[(games['positive_ratio'] >= 50) & (games['user_reviews'] >= 30) & (games['date_release'] > twenty_years_ago)]

content_recom = pd.merge(filtered_games, metadata, on='app_id')
relevant_cols = content_recom[['app_id', 'title', 'tags']]

all_tags = ','.join(relevant_cols['tags']).split(',')
tag_counts = {}
for tag in all_tags:
    if tag in tag_counts:
        tag_counts[tag] += 1
    else:
        tag_counts[tag] = 1

popular_tags = {tag: count for tag, count in tag_counts.items() if count > 2000}
sorted_data = sorted(popular_tags.items(), key=lambda x: x[1], reverse=True)
df_sorted_data_tag = pd.DataFrame(sorted_data, columns=['Tag', 'Count'])
# CSV dosyası olarak çıkarma
#df_sorted_data_tag.to_csv('top_15_tag_counts.csv', index=False)
df_sorted_data_tag.to_parquet('top_15_tag_counts.parquet', index=False)


for item in sorted_data[:15]:
    print(item)

for tag in popular_tags:
    relevant_cols[tag] = relevant_cols['tags'].str.contains(tag).astype(int)
relevant_cols.columns = relevant_cols.columns.str.strip()
relevant_cols.loc[:, ~relevant_cols.columns.isin(['app_id','title', 'description', 'tags']) & ~relevant_cols.columns.isin(popular_tags)] = 0
relevant_cols.info()
relevant_cols.isnull().any()

relevant_cols = relevant_cols.loc[:, ~relevant_cols.columns.duplicated()]

drop_columns=['2D', '3D',  'Anime',  'Co-op', 'Colorful',  'Comedy',
       'Cute', 'Difficult', 'Early Access',
       'Exploration',  'Family Friendly', 'Fantasy',
       'Female Protagonist', 'First-Person', 'Free to Play', 'Funny', 'Great Soundtrack', 'Horror',
        'Open World',  'Pixel Graphics',
       'Platformer',
       'Relaxing', 'Retro',
       'Sci-fi', 'Shooter',"Violent",
       'app_id', 'tags', 'title']
features=relevant_cols.drop(columns=drop_columns, axis=1)
features.info()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', NearestNeighbors(n_neighbors=5))
])
pipeline.fit(features)


def get_recommendations(app_id, data, model_pipeline):
    game_index = data.loc[data['app_id'] == app_id].index[0]

    game_features = features.iloc[[game_index]]

    _, indices = model_pipeline.named_steps['knn'].kneighbors(
        model_pipeline.named_steps['scaler'].transform(game_features), n_neighbors=2)

    recommended_index = indices[0][1]

    return data.iloc[recommended_index]['app_id']


relevant_cols.to_parquet('data.parquet', index=False)


app_id = relevant_cols.sample(1)['app_id'].values[0]
recommendations = get_recommendations(app_id, relevant_cols, pipeline)

game_name = relevant_cols.loc[relevant_cols['app_id'] == app_id, ['title']].values[0]
recom_game= relevant_cols.loc[relevant_cols['app_id'] == recommendations, ['title']].values[0]

print(f"gamename: '{game_name}' recom_gamename: {recom_game}")

joblib.dump(pipeline, 'knn_game_recommender_pipeline.pkl')


# Content-Based Filtering

#content_based için veriseti oluşturma
metadata_cont = pd.read_json('games_metadata.json', lines=True)
metadata_cont['description'] = metadata_cont['description'].apply(clean_text)
metadata_cont['description'] = metadata_cont['description'].apply(lambda x: np.nan if x == '' else x)
metadata_cont['description'].isnull().sum()
metadata_cont.dropna(inplace=True)
games = pd.read_csv('games.csv')
content_recom = pd.merge(games, metadata_cont, on='app_id')
content_recom.info()
filtered_data = content_recom[content_recom['positive_ratio'] > 85]
relevant_cols2 = filtered_data[['app_id', 'title', 'description']]



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# TF-IDF
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(relevant_cols2['description'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

joblib.dump((tfidf_vectorizer, cosine_similarities), 'content_recommendation_model.pkl')


relevant_cols2.to_parquet("description_data.parquet")
games = pd.read_csv('games.csv')
metadata = pd.read_json('games_metadata.json', lines=True)
metadata['description'] = metadata['description'].apply(lambda x: np.nan if x == '' else x)
metadata['description'].isnull().sum()
metadata.dropna(inplace=True)
metadata.to_parquet("tag_data.parquet")
games.to_parquet("games.parquet")