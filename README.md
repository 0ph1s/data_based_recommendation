# 🎬 Movie Recommendation System 🎥🍿  

This repository contains a **movie recommendation system** built with Python. The script uses **TF-IDF vectorization** and **cosine similarity** to suggest similar movies based on user input.  

## 📌 Features  
- Reads movie data from a CSV file.  
- Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to process textual features like **genres, keywords, cast, and director**.  
- Computes **cosine similarity** to find the most relevant movie recommendations.  
- Implements **error handling** for missing files, empty datasets, and incorrect user input.  

## 📂 Data  
The script expects a `movies.csv` file containing movie details, including:

- `title`  
- `genres`  
- `keywords`  
- `tagline`  
- `cast`  
- `director`  

## 🔧 Technologies  
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Difflib  

## 🚀 How to Use  
1. Clone the repository.  
2. Install dependencies:
    
   ```bash
   pip install numpy pandas scikit-learn
   ```
   
4. Ensure the `movies.csv` file is present.  
5. Run the script and input a movie title to receive recommendations.  

## 📊 Example  

```
Digite o nome do filme: Inception  
Quantas recomendações deseja receber? 5  

Filmes sugeridos para você:  
1. Interstellar  
2. The Prestige  
3. Memento  
4. The Dark Knight  
5. Tenet  

Deseja continuar? sim
```

Feel free to **modify, explore, and contribute**! 🎬✨
