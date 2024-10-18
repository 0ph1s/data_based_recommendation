import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    # Read Data
    df = pd.read_csv('movies.csv')
    # Print Data
    ##print(df.head(5))
except FileNotFoundError:
    print("O arquivo 'movies.csv' não foi encontrado.")
except pd.errors.EmptyDataError:
    print("O arquivo está vazio.")
except pd.errors.ParserError:
    print("Erro ao ler o arquivo CSV.")

# Data information
##print(df.info())

# Selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
##print(selected_features)

# Troca valores null por strings null
for feature in selected_features:
    df[feature] = df[feature].fillna('')

combined_features = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
##print(combined_features)

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

##print(feature_vectors)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors, feature_vectors)

##print(similarity)

# creating a list with all the movie names given in the dataset
list_of_all_titles = df['title'].tolist()

movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = df['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = df[df.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Filmes sugeridos para você: \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = df[df.index==index]['title'].values[0]
    if (i < 6):
        print(i, '.',title_from_index)
        i+=1



