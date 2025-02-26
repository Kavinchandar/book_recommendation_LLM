---
title: book_recommendation_LLM
app_file: gradio_dashboard.py
sdk: gradio
sdk_version: 5.15.0
---

Building a sematic book recommendation application powered by LLMs

Users can write prompts to get book recommendations of what they have read or other custom user prompts and then filter by category and tone.

how it works:

1. Applied LLMs to retrieve data from a Database based on user prompts model: (Open AI API)

2. Used zero-shot text classification to normalize categories model: (facebook/bart-large-mnli)

3. Used sentiment analysis to classify books by tone model: (j-hartmann/emotion-english-distilroberta-base)


source video: https://www.youtube.com/watch?v=Q7mS1VHm3Yw


<img width="1261" alt="image" src="https://github.com/user-attachments/assets/3b22c49d-9beb-4525-8b34-d93cbbf18e44" />

