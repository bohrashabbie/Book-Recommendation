import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('book_pivot.pkl', 'rb') as file:
    book_pivot = pickle.load(file)
    
with open('books_name.pkl', 'rb') as file:
    books_name = pickle.load(file)
    
with open('final_ratings.pkl', 'rb') as file:
    final_ratings = pickle.load(file)


            
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance,suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
    
    recommend = []
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            recommend.append(j) 
    return recommend          
            
# book_name = 'Winter Solstice'
# recommend_book(book_name)

# Streamlit UI
def main():
    st.title("Book Recommendation System")
    st.write("Welcome to the Book Recommendation System! Enter a book name and get recommendations.")

    # User input for book name
    book_name = st.text_input("Enter a book name:", "")

    if st.button("Get Recommendations"):
        if book_name:
            # Displaying recommendations
            recommendations = recommend_book(book_name)
            st.subheader("Top 5 Recommended Books:")
            for book in recommendations:
                st.write(f"- {book}")
        else:
            st.warning("Please enter a book name.")

if __name__ == "__main__":
    main()
