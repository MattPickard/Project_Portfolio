# RAG Techniques to Query my Grandfather's Memoir
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/family_photo.jpg" alt="Family Photo" style="width: 60%;">
</p>
<p align="center">
<em>I am being held by my grandfather on the right.</em>
</p>

## Table of Contents
* [Implementation Resources](#implementation-resources)
* [Introduction](#introduction)
* [Data](#data)
* [Basic RAG](#basic-rag)

<a name="implementation-resources"></a>
## Implementation Resources
* [https://github.com/NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
* [https://github.com/microsoft/GraphRAG](https://github.com/microsoft/GraphRAG)

<a name="introduction"></a>
## Introduction

This project was an exploration of various RAG (Retrieval-Augmented Generation) techniques applied to a dataset that is particularly meaningful to me - a memoir written by my grandfather. The hope was to demonstrate and improve my intuitions and understandings of RAG and NLP (Natural Language Processing) pipelines. In total, I built and evaluated four pipelines with the goal of comparing various techniques and architectures. I started with a simple RAG pipeline to use as a baseline, then worked on an ensemble of techniques including **reranking**, **content enrichment window**, and **query rewriting**. Finally, I implemented and evaluated Microsoft's GraphRAG due to GraphRAG's popularity in the RAG community as an architecture.

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/success_rates.png" alt="Success Rates" style="width: 70%">
</p>
<p align="center">
  <em>The evaluation success rate of each pipeline.</em>
</p>

<a name="data"></a>
## Data

The dataset is a memoir written by my grandfather called "My Life Story". The memoir is about 42,000 words which I split into 10 chapters. Each chapter was saved as a separate PDF file so I could practice and implement pipelines that preprocess PDF documents.

<a name="basic-rag"></a>
## Basic RAG  
Code: [Basic RAG Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/basic_rag.ipynb)

This first pipeline, built with LangChain, is an implementation of a simple RAG architecture without any additional techniques. At its core, it consists of storing chunks of the text in a vector database via encodings and then querying the database, retrieving the two most similar chunks to craft a response. To preprocess the text, I used PyPDFLoader to convert the PDF files into text, then used RecursiveCharacterTextSplitter to split the text into 1000 character chunks with 200 character overlaps. I appended the chapter source to the end of each chunk for better retrieval. Each chunk was embedded using the OpenAI Embeddings API and converted into a vector database using FAISS (Facebook AI Similarity Search).
When queried, the database retrieves the top two most similar chunks and then uses OpenAI's GPT-4o-mini model to generate a response. The following prompt template was provided to the model:
```
You are querying a memoir called "My Life Story" written by George Shambaugh.
    For the question below, provide a concise but sufficient answer. If you don't know, only write "The RAG retrieval was unable to provide sufficient context":
    {context}
    Question
    {question}
```

Evaluation and Insights:

**Success Rate: 60%**

During evaluation, two major issues were identified that were limiting the pipeline's performance:

1. The chunks did not carry over context across pages, which is a result of PyPDFLoader's default behavior of splitting the documents into pages.
2. Relying solely on the top two most similar chunks without any other techniques led to many situations where the pipeline was unable to retrieve the proper context to answer the question.

I worked to address both of these issues in the following implementation of the RAG Ensemble pipeline. 

During the testing phase, I also observed that OpenAI's GPT-4o-mini model consistently failed to include sources in its responses, despite it being prompted and the sources being present in the provided context. This behavior persisted even after modifying the prompts and positioning the sources within the context. It is possible that this is a characteristic of the model or the model following an OpenAI policy, and it may be beneficial to explore alternative models that are more accommodating to such requests. If I were to continue this project, testing different models would be a logical next step.

<a name="rag-ensemble"></a>
## RAG Ensemble  
Code: [RAG Ensemble Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/rag_ensemble.ipynb)
