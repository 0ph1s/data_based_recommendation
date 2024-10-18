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

while True:
    nome_do_filme = input('\nDigite o nome do filme: ')

    while True:
        try:
            qnt_recomendacao = int(input('\nQuantas recomendações deseja receber? '))
            if qnt_recomendacao > 0:
                break
            else:
                print("\nPor favor, insira um número positivo.")
        except ValueError:
            print("\nEntrada inválida. Por favor, insira um número inteiro.")

    encontrar_mais_proximo = difflib.get_close_matches(nome_do_filme, lista_de_todos_os_titulos)

    if not encontrar_mais_proximo:
        print("\nFilme não encontrado. Tente novamente.")
        continue

    titulo_mais_proximo = encontrar_mais_proximo[0]
    indice_do_filme = df[df.title == titulo_mais_proximo]['index'].values[0]

    pontuacao_similaridade = list(enumerate(similaridade[indice_do_filme]))
    filmes_similares_ordenados = sorted(pontuacao_similaridade, key=lambda x: x[1], reverse=True)

    print('Filmes sugeridos para você: \n')

    i = 1
    for filme in filmes_similares_ordenados:
        indice = filme[0]
        titulo_do_indice = df[df.index == indice]['title'].values[0]
        if i <= qnt_recomendacao:
            print(i, '.', titulo_do_indice)
            i += 1
        if i > qnt_recomendacao:
            break

    response = input("\nDeseja continuar? ")

    if response.lower() == 'sim':
        continue
    else:
        print("\nObrigado pela utilização!\n")
        break

