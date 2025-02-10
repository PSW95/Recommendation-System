import streamlit as st 
import pandas as pd

import difflib as dl


import time


df=pd.read_csv('Top3000_imdb_indian_movies.csv')

data_features=df[['Genre','Plot_summary','Cast_stars','Crew_dir']].fillna('')
x=data_features['Genre']+' '+data_features['Plot_summary']+' '+data_features['Cast_stars']+' '+data_features['Crew_dir']

from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()
x=tf.fit_transform(x)

from sklearn.metrics.pairwise import cosine_similarity

cs=cosine_similarity(x)

movie_title=df['Title'].tolist()

import difflib as dl

st.markdown("<h1 style='text-align: center; color: white;'>🎥 Hindi Movie Recommendation System 🎥</h1>", unsafe_allow_html=True)


# st.header("Hindi Movies Recommendation")
movie_name=st.text_input("Enter the name of movie " ,placeholder="E.g., Dangal, Sholay, 3 Idiots", help="Type a movie name and press on get recommendation")

# Recommendation button
if st.button("Get Recommendations"):
    # Find the closest match to the movie name
    recommendation = dl.get_close_matches(movie_name,movie_title)
    if recommendation:
        close_match=recommendation[0]
        index_number=df[df.Title == close_match]['Movie_id'].index[0]
        recommendation_score=list(enumerate(cs[index_number]))
        sorted_similar=sorted(recommendation_score,key= lambda x:x[1],reverse=True)
        print("these are the top 10 movies recommend for you\n")
        i=1

        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            progress_bar.progress(percent_complete + 1)

        for movie in sorted_similar:
            index=movie[0]
            title_from_idx=df[df.index==index]['Title'].values[0]
            if (i<11):
                st.write(i," ",title_from_idx ,'🎬')
                i+=1



    
        
    else:
        st.write("No similar movies found. Please try a different movie name.")

