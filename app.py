import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

st.set_page_config(page_title='Recommendation System', layout="wide", initial_sidebar_state="auto", menu_items=None) #, page_icon='atom.png'

hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)


c1,c2,c3 = st.columns([2,4,2], gap='large')

c2.markdown("<h1 style='text-align: center;'><font face='High Tower Text'>Movie Recommendation System </font></h1>", unsafe_allow_html=True)
c2.markdown("<h3 style='text-align: center;'><font face='High Tower Text'> By Mayur Shrotriya </font></h3>", unsafe_allow_html=True)

st.markdown("***", unsafe_allow_html=True)

df_movies = pd.read_csv('data/tmdb_5000_movies.csv')

tfidf = TfidfVectorizer(stop_words='english')
df_movies['overview'].fillna("", inplace=True)
tfidf_matrix = tfidf.fit_transform(df_movies['overview'])

cosine_sim = linear_kernel(tfidf_matrix , tfidf_matrix )

indices = pd.Series(df_movies.index , index = df_movies['original_title']).drop_duplicates()

def get_recommendations(title, cosine_sim = cosine_sim, thresh = 0.1):
    idx = indices[title]
    sim_scores = enumerate(cosine_sim[idx])
    sim_scores = sorted(sim_scores, key =lambda x:x[1], reverse=True)
    sim_scores = list(filter( lambda x: True if x[1] > thresh else False, sim_scores))
    sim_scores = sim_scores[1:11]
    sim_index = [i[0] for i in sim_scores]
    return df_movies[ "original_title" ].iloc[sim_index].values , df_movies[ "id" ].iloc[sim_index].values

import requests
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

    
c1,c2,c3,c4,c5 = st.columns([1,2,2,2,1], gap='large')

searchFor = c2.selectbox('Select your movie Title', [''] + sorted(df_movies['title'].unique()))

thresh = c3.number_input('Enter minimum percentage (%) similarity you want',min_value=0, max_value=100, value=10)

thresh = thresh/100

if c2.button('Get Recommendations') and searchFor != '':
    
    c2.write('The recommeded movies are :')
    
    rc = get_recommendations(searchFor, thresh=thresh)
    
    poster_urls = []    
    for id in rc[1]:
        poster_urls.append(fetch_poster(id))
    for n, item,poster in zip(range(len(rc[0])),rc[0],poster_urls):
        if n==0 or n%2==0:
            st.markdown('***')
            c1,c2,c3,c4,c5,c6 = st.columns([1,1,2,1,2,1], gap='large')
        if n%2==0:
            c2.write(str(n+1)+'. '+item)
            c3.image(poster,width=300)
        else:
            c4.write(str(n+1)+'. '+item)
            c5.image(poster,width=300)
    st.markdown('***')
            
    
