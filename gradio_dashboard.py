import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where( 
    books["large_thumbnail"].isna(),
    "no_cover.jpg", 
    books["large_thumbnail"],
)

# Path to store ChromaDB
CHROMA_DB_PATH = "chroma_db"

# Load or Create ChromaDB
if os.path.exists(CHROMA_DB_PATH):
    db_books = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())
    print("Loaded existing Chroma vector database.")
else:
    print("Creating a new Chroma vector database...")
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
    db_books.persist()
    print("Chroma vector database created and saved.")

# REMOVED the duplicate database creation that was overwriting the persistent DB

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    recommendations = db_books.similarity_search_with_score(query, k=initial_top_k)
    print(f"Found {len(recommendations)} initial matches")
    
    book_ids = [int(recommendation[0].page_content.strip('"').split()[0]) for recommendation in recommendations]
    book_recommendations = books[books["isbn13"].isin(book_ids)]
    
    # Case-insensitive comparison for category
    if category and category.upper() != 'ALL':
        book_recommendations = book_recommendations[book_recommendations['simple_categories'] == category]
        
    # Limit results after filtering
    book_recommendations = book_recommendations.head(final_top_k)

    # Apply tone filtering
    if tone and tone.upper() != 'ALL':
        if tone == 'Happy':
            book_recommendations.sort_values(by="joy", ascending=False, inplace=True)
        elif tone == "Surprising" or tone == "surprising":
            book_recommendations.sort_values(by="surprise", ascending=False, inplace=True)
        elif tone == "Angry" or tone == "angry":
            book_recommendations.sort_values(by="anger", ascending=False, inplace=True)
        elif tone == "Suspenseful":
            book_recommendations.sort_values(by="fear", ascending=False, inplace=True)
        elif tone == "Sad":
            book_recommendations.sort_values(by="sadness", ascending=False, inplace=True)
        elif tone == 'Disgust':
            book_recommendations.sort_values(by="disgust", ascending=False, inplace=True)

    print(f"Returning {len(book_recommendations)} final recommendations")
    return book_recommendations


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"  # Fixed f-string
        elif len(authors_split) > 2:
            authors_str = f"{','.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    print(f"Generated {len(results)} gallery items")
    return results


# Standardize capitalization in dropdown options
categories = ['All'] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad", "Disgust"]  # Fixed capitalization

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# Semantic Book Recommender')

    with gr.Row():
        user_query = gr.Textbox(
            label='Please enter a description of a book:', 
            placeholder='e.g., A story about forgiveness'
        )
        category_dropdown = gr.Dropdown(
            choices=categories, 
            label="Select a category:", 
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, 
            label="Select the tone:", 
            value="All"
        )
        submit_button = gr.Button('Find recommendations')

    gr.Markdown('## Recommendations')

    output = gr.Gallery(label='Recommended Books', columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )


if __name__ == "__main__":
    dashboard.launch()