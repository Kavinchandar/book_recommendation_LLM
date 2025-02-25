import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where( 
    books ["large_thumbnail"].isna(),
    "no_cover.jpg", 
    books["large_thumbnail"],
)


raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
print(documents[0])
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retreive_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    recommendations = db_books.similarity_search_with_score(query,k=initial_top_k)
    book_ids = [int(recommendations.page_content.strip('"').split()[0]) for recommendations in recommendations]
    book_recommendations = books[books["isbn13"]].isin(book_ids).head(final_top_k)

    if category != 'ALL':
        book_recommendations = book_recommendations[book_recommendations['simple_categories'] == category][:final_top_k]
    else:
        book_recommendations = book_recommendations.head(final_top_k)

    if tone == 'Happy':
        book_recommendations.sort_values(by="joy",ascending=False,inplace=True)
    elif tone == "Surprising":
        book_recommendations.sort_values(by="surprise",ascending=False, inplace=True)
    elif tone == "Angry":
        book_recommendations.sort_values (by="anger",ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recommendations.sort_values (by="fear",ascending=False, inplace=True)
    elif tone == "Sad":
        book_recommendations.sort_values (by="sadness",ascending=False, inplace=True)
    elif tone == 'Disgust':
        book_recommendations.sort_values (by="disgust",ascending=False, inplace=True)

    return book_recommendations