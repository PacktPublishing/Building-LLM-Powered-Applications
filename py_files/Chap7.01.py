import os
from dotenv import load_dotenv

import numpy as np

# Row = user
# Col = movie

user_movie_matrix = np.array([
    [4, 0, 5, 0],
    [0, 3, 0, 2],
    [5, 4, 0, 3]
])

""" 
    Matrix factorization aims to break down this matrix into two matrices: 
    one for users and another for movies, 
    with a reduced number of dimensions (latent factors). 
    These latent factors could represent attributes like genre preferences or specific movie characteristics. 
    By multiplying these matrices, you can predict the missing ratings and recommend movies that the users might enjoy.

    

on effectue une décomposition en valeurs singulières (SVD) d'une matrice utilisateur-film pour réduire la dimensionnalité et reconstruire une matrice "approximée" en utilisant un nombre réduit de "facteurs latents". Voici une explication détaillée :

### 1. **Matrice utilisateur-film (`user_movie_matrix`)** :
La matrice initiale représente les notes données par des utilisateurs à des films. Les lignes correspondent aux utilisateurs et les colonnes aux films. Une valeur de 0 indique qu'un utilisateur n'a pas noté un film.

```python
user_movie_matrix = np.array([
    [4, 0, 5, 0],
    [0, 3, 0, 2],
    [5, 4, 0, 3]
])
```

- Par exemple, l'utilisateur 1 (ligne 1) a donné une note de 4 au film 1 (colonne 1) et de 5 au film 3 (colonne 3), mais n'a pas noté les films 2 et 4.

### 2. **Décomposition SVD (`np.linalg.svd`)** :
La décomposition en valeurs singulières (SVD) est une technique de réduction de dimension qui factorise une matrice en trois composants : `U`, `s`, et `V` tels que :

Matrice originale = U . s . V^T

- `U`: Matrice des utilisateurs (relation des utilisateurs avec les "facteurs latents").
- `s`: Valeurs singulières (composantes principales ou poids associés aux facteurs latents).
- `V`: Matrice des films (relation des films avec les "facteurs latents").

```python
U, s, V = np.linalg.svd(user_movie_matrix, full_matrices=False)
```

Ici, `full_matrices=False` signifie que seules les dimensions minimales nécessaires sont conservées.

### 3. **Sélection des facteurs latents** :
La variable `num_latent_factors` permet de choisir le nombre de "facteurs latents" que l'on souhaite conserver après la décomposition. Cela contrôle le niveau de réduction de la dimensionnalité.

```python
num_latent_factors = 2
```

Dans ce cas, on choisit de garder seulement 2 facteurs latents.

### 4. **Reconstruction de la matrice** :
La matrice d'origine est reconstruite en utilisant uniquement les 2 facteurs latents sélectionnés.

```python
reconstructed_matrix = U[:, :num_latent_factors] @ np.diag(s[:num_latent_factors]) @ V[:num_latent_factors, :]
```

Ici :
- `U[:, :num_latent_factors]` conserve les 2 premières colonnes de `U`.
- `s[:num_latent_factors]` conserve les 2 premières valeurs singulières (sous forme de matrice diagonale).
- `V[:num_latent_factors, :]` conserve les 2 premières lignes de `V`.

### 5. **Suppression des valeurs négatives** :
SVD peut générer des valeurs négatives dans la matrice reconstruite, ce qui n'est pas pertinent pour une matrice de notation de films. On remplace donc les valeurs négatives par 0.

```python
reconstructed_matrix = np.maximum(reconstructed_matrix, 0)
```

### 6. **Affichage de la matrice reconstruite** :
Enfin, on affiche la matrice reconstruite après réduction de la dimension.

```python
print("Reconstructed Matrix:")
print(reconstructed_matrix)
```

### Exemple de sortie :
La matrice reconstruite sera une approximation de la matrice initiale, mais avec une dimensionnalité réduite et des informations compressées.

### But :
Cette technique est utilisée dans les systèmes de recommandation pour réduire la matrice utilisateur-film en un espace latent plus petit. Cela permet de mieux généraliser les relations entre les utilisateurs et les films en utilisant moins de données, et d'effectuer des recommandations basées sur des modèles factorisés.

### Exemple pratique :
- **Utilisateurs** : 3
- **Films** : 4
- **Notes** : de 0 (non noté) à 5

La réduction de dimension permet de capturer les similarités entre utilisateurs ou films et d'améliorer la prédiction des notes pour les films non notés.


Les **facteurs latents** représentent des concepts ou des dimensions sous-jacentes qui ne sont pas explicitement visibles dans les données d'origine, mais qui influencent les relations entre les utilisateurs et les objets (films, produits, etc.).

### Contexte dans une matrice utilisateur-film
Prenons un exemple simplifié :

1. **Matrice utilisateur-film** : Suppose que tu as une matrice où chaque ligne représente un utilisateur et chaque colonne représente un film, et les valeurs sont les notes que les utilisateurs donnent aux films. La matrice pourrait ressembler à ceci :

   |       | Film 1 | Film 2 | Film 3 | Film 4 |
   |-------|--------|--------|--------|--------|
   | User 1|   4    |   0    |   5    |   0    |
   | User 2|   0    |   3    |   0    |   2    |
   | User 3|   5    |   4    |   0    |   3    |

2. **Facteurs latents** : Derrière les notes que les utilisateurs donnent aux films, il y a des préférences ou des goûts qui ne sont pas directement visibles. Par exemple, certains utilisateurs aiment les films d'action, d'autres préfèrent les comédies, et d'autres encore aiment les films romantiques.

   - **Facteurs latents pour les utilisateurs** : Ces préférences sont les **facteurs latents** qui influencent les notes des utilisateurs. Par exemple, un facteur latent pourrait être l'intérêt pour les films d'action, un autre pour les films romantiques, etc.
   - **Facteurs latents pour les films** : De même, chaque film peut être décrit en fonction de ces mêmes facteurs latents. Par exemple, un film pourrait avoir un score élevé pour le facteur "action" et un score faible pour le facteur "comédie".

### Comment les facteurs latents fonctionnent dans SVD

Lorsque tu appliques la décomposition en valeurs singulières (SVD) à la matrice utilisateur-film, tu divises la matrice en trois parties : 
1. **U** : Matrice des utilisateurs par rapport aux facteurs latents (préférences des utilisateurs).
2. **S** : Les valeurs singulières, qui représentent l'importance de chaque facteur latent.
3. **V** : Matrice des films par rapport aux facteurs latents (caractéristiques des films).

L'idée est que, même si tu n'as pas une représentation explicite des préférences des utilisateurs et des caractéristiques des films, SVD trouve ces **facteurs latents** de manière mathématique. Ce sont des dimensions "cachées" qui capturent les aspects communs entre utilisateurs et films.

### Exemple simplifié des facteurs latents
Disons qu'il y a deux **facteurs latents** :
- **Facteur 1** : Intérêt pour les films d'action.
- **Facteur 2** : Intérêt pour les comédies romantiques.

Chaque utilisateur pourrait avoir un score pour chacun de ces facteurs, représentant à quel point il aime les films d'action ou de comédie romantique. De même, chaque film aurait des scores pour ces mêmes facteurs, représentant s'il est plus d'action ou de comédie romantique.

- **Utilisateur 1** pourrait avoir un score élevé pour "action" et faible pour "comédie romantique".
- **Film 1** pourrait être un film d'action avec un score élevé pour "action" et faible pour "comédie romantique".

En combinant ces scores, tu peux prédire quelle note un utilisateur pourrait donner à un film, même s'il ne l'a pas encore regardé. 

### Résumé
Les **facteurs latents** sont des dimensions cachées qui expliquent pourquoi un utilisateur aime ou n'aime pas un film (ou tout autre objet). Ils sont découverts grâce à des méthodes comme la décomposition SVD et permettent de comprendre les relations sous-jacentes dans les données.




"""	


# Apply SVD Singular Value Decomposition
# U singular Vector
# s Singular value

U, s, V = np.linalg.svd(user_movie_matrix, full_matrices=False)

# Number of latent factors (you can choose this based on your preference)
num_latent_factors = 2

# Reconstruct the original matrix using the selected latent factors
# le @ effetue le produit matriciel

reconstructed_matrix = U[:, :num_latent_factors] @ np.diag(s[:num_latent_factors]) @ V[:num_latent_factors, :]

# Replace negative values with 0
reconstructed_matrix = np.maximum(reconstructed_matrix, 0)
print("Reconstructed Matrix:")
print(reconstructed_matrix)



