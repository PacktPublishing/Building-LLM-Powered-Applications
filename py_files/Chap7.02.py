import os
import pandas as pd
import tiktoken
import openai
import lancedb

"""
    The `ast` module helps Python applications to process trees of the Python abstract syntax grammar. 
"""

import ast

openai.api_key = os.environ["OPENAI_API_KEY"]

#md = pd.read_csv('movies_metadata.csv', dtype={10:str})  
md = pd.read_csv('movies_metadata.csv', dtype={'popularity':str})  

#print(md.columns[10])
#print(md.iloc[:100,10])

# Convert string representation of dictionaries to actual dictionaries
md['genres'] = md['genres'].apply(ast.literal_eval)

# Transforming the 'genres' column
md['genres'] = md['genres'].apply(lambda x: [genre['name'] for genre in x])

"""
3. Utilisation de lambda pour transformer les genres :
md['genres'] = md['genres'].apply(lambda x: [genre['name'] for genre in x])
Cette ligne applique une fonction lambda sur chaque élément de la colonne 'genres'. Voici comment cela fonctionne :
apply() : La méthode .apply() est utilisée pour appliquer une fonction à chaque élément de la colonne Pandas.
lambda x: [...] : Une lambda est une fonction anonyme (sans nom) définie directement dans une expression. Ici, x représente chaque élément de la colonne 'genres'.
[genre['name'] for genre in x] : Il s'agit d'une liste en compréhension (list comprehension) qui parcourt chaque élément genre dans la liste x (chaque x étant une liste de dictionnaires représentant des genres) et extrait la valeur associée à la clé 'name'.

Exemple concret :
Imaginons que pour un film, l'élément x dans la colonne 'genres' soit :

[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]

La fonction lambda va parcourir cette liste et extraire uniquement les noms des genres, produisant la liste suivante :
['Action', 'Adventure']


print(md['genres'] )


Next, we merge the vote_average and vote_count columns into a single column, which is the weighted ratings with respect to the number of votes. 
I’ve also limited the rows to the 95th percentile of the number of votes, so that we can get rid of minimum vote counts to prevent skewed results:
"""

# Calculate weighted rate (IMDb formula)
def calculate_weighted_rate(vote_average, vote_count, min_vote_count=10):
    return (vote_count / (vote_count + min_vote_count)) * vote_average + (min_vote_count / (vote_count + min_vote_count)) * 5.0

# Minimum vote count to prevent skewed results
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')

min_vote_count = vote_counts.quantile(0.95)

# Create a new column 'weighted_rate'
md['weighted_rate'] = md.apply(lambda row: calculate_weighted_rate(row['vote_average'], row['vote_count'], min_vote_count), axis=1)


#Next, we create a new column called combined_info where we are going to merge all the elements that will be provided as context to the LLMs. Those elements are the movie title, overview, genres, and ratings:

md = md.dropna()

md_final = md[['genres', 'title', 'overview', 'weighted_rate']].reset_index(drop=True)

#print(md_final.columns)

md_final.head()

md_final['combined_info'] = md_final.apply(lambda row: f"Title: {row['title']}. Overview: {row['overview']} Genres: {', '.join(row['genres'])}. Rating: {row['weighted_rate']}", axis=1).astype(str)

#print(md_final.head())
#print(md_final.columns)

#We tokenize the movie combined_info so that we will get better results while embedding:

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.embeddings.create(input = [text], model=model).data[0].embedding


"""
cl100k_base is the name of a tokenizer used by OpenAI’s embeddings API. 
A tokenizer is a tool that splits a text string into units called tokens, 
which can then be processed by a neural network. 
Different tokenizers have different rules and vocabularies for how to split the text and what tokens to use.

The cl100k_base tokenizer is based on the byte pair encoding (BPE) algorithm,
 which learns a vocabulary of subword units from a large corpus of text. 
 The cl100k_base tokenizer has a vocabulary of 100,000 tokens, which are mostly common words and word pieces, 
 but also include some special tokens for punctuation, formatting, and control. 
 It can handle texts in multiple languages and domains, and can encode up to 8,191 tokens per input.
"""

embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
embedding_model = "text-embedding-ada-002"

max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
encoding = tiktoken.get_encoding(embedding_encoding)
# omit reviews that are too long to embed
md_final["n_tokens"] = md_final.combined_info.apply(lambda x: len(encoding.encode(x)))
md_final = md_final[md_final.n_tokens <= max_tokens]

#print(md_final.columns)

# omit reviews that are too long to embed
len(md_final)

print(md_final['genres']);


#print(md_final.head())

md_final["embedding"] = md_final.overview.apply(lambda x: get_embedding(x, model=embedding_model))

print(md_final.head())

md_final.rename(columns = {'embedding': 'vector'}, inplace = True)
md_final.rename(columns = {'combined_info': 'text'}, inplace = True) # 

def convert_list_to_dict(genre_list):
    return {'metadata': genre_list}

md_final['genres_dict'] = md_final['genres'].apply(convert_list_to_dict)
md_final.rename(columns = {'genres_dict': 'metadata'}, inplace = True) # 

md_final.to_pickle('movies.pkl')  # Pickle (serialize) object to file.

uri = "data/sample-lancedb"
db = lancedb.connect(uri)

db.drop_table("movies", ignore_missing=True)
table = db.create_table("movies", md_final)

print("done")
