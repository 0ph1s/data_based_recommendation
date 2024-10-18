import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    df = pd.read_csv('movies.csv')
    ##print(df.head(5))
except FileNotFoundError:
    print("O arquivo 'movies.csv' não foi encontrado.")
except pd.errors.EmptyDataError:
    print("O arquivo está vazio.")
except pd.errors.ParserError:
    print("Erro ao ler o arquivo CSV.")

##print(df.info())

recursos_selecionados = ['genres','keywords','tagline','cast','director']
##print(recursos_selecionados)

for recurso in recursos_selecionados:
    df[recurso] = df[recurso].fillna('')

recursos_combinados = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
##print(recursos_combinados)

vetorizador = TfidfVectorizer()

vetores = vetorizador.fit_transform(recursos_combinados)

##print(vetores)

similaridade = cosine_similarity(vetores, vetores)

##print(similaridade)

lista_de_todos_os_titulos = df['title'].tolist()

nome_do_filme = input('Digite o nome do filme: ')

encontrar_mais_proximo = difflib.get_close_matches(nome_do_filme, lista_de_todos_os_titulos)

titulo_mais_proximo = encontrar_mais_proximo[0]

indice_do_filme = df[df.title == titulo_mais_proximo]['index'].values[0]

pontuacao_similaridade = list(enumerate(similaridade[indice_do_filme]))

filmes_similares_ordenados = sorted(pontuacao_similaridade, key = lambda x:x[1], reverse = True) 

print('Filmes sugeridos para você: \n')

i = 1

for filme in filmes_similares_ordenados:
    indice = filme[0]
    titulo_do_indice = df[df.index==indice]['title'].values[0]
    if (i < 6):
        print(i, '.',titulo_do_indice)
        i+=1
