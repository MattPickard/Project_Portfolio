# RAG Techniques to Query my Grandfather's Memoir
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/family_photo.jpg" alt="Family Photo" style="width: 60%;">
  <em>A family photo of me being held by my grandfather.</em>
</p>

## Table of Contents

* [Introduction](#introduction)
* [The Data](#data)
* [RAG Architectures](#rag-architectures)


<a name="introduction"></a>
## Introduction

This project is an exploration of various RAG (Retrieval-Augmented Generation) techniques applied to a dataset that is particularly meaningful to me - a memoir written by my grandfather. The hope is to show off and improve my intuitions and understandings of RAG and NLP (Natural Language Processing) pipelines. In total, I built and evaluated four pipelines with the goal of comparing various techniques and architectures. I started with a simple RAG pipeline to use as a baseline, then worked on an enseble of techniques including **reranking**, **content enrichment window**, and **query rewriting**. Finally, I implimented and evaluated Microsoft's GraphRAG due to GraphRAG's popularity in the RAG community as an architecture.

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/success_rates.png" alt="Success Rates" style="width: 70%; margin: 0; padding: 0;">
</p>
<p align="center" style="margin: 0; padding: 0;">
  <em>The success rate of each pipeline.</em>
</p>

<a name="data"></a>
## The Data

The dataset is a memoir written by my grandfather called "My Life Story." The memoir is about 42,000 words which I split into 10 chapters. Each chapter was saved as a PDF file so I could practice and and impliment pipelines that could preprocess PDF documents.

<a name="rag-architectures"></a>
## RAG Architectures

There are two RAG architectures that will be used in this project. The first is a simple RAG architecture that will be used to query the memoir. The second is a more complex RAG architecture that will be used to query the memoir.


