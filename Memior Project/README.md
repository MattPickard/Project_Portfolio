# RAG Techniques to Query my Grandfather's Memoir
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/family_photo.jpg" alt="Family Photo" style="width: 60%;">
</p>
<p>
<em>I'm being held by my grandfather on the right.</em>
</p>

## Table of Contents

* [Introduction](#introduction)
* [The Data](#data)
* [RAG Architectures](#rag-architectures)

<a name="implimentation-resources"></a>
## Implimentation Resources
* [https://github.com/NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
* [https://github.com/microsoft/GraphRAG](https://github.com/microsoft/GraphRAG)

<a name="introduction"></a>
## Introduction

This project was an exploration of various RAG (Retrieval-Augmented Generation) techniques applied to a dataset that is particularly meaningful to me - a memoir written by my grandfather. The hope was to show off and improve my intuitions and understandings of RAG and NLP (Natural Language Processing) pipelines. In total, I built and evaluated four pipelines with the goal of comparing various techniques and architectures. I started with a simple RAG pipeline to use as a baseline, then worked on an enseble of techniques including **reranking**, **content enrichment window**, and **query rewriting**. Finally, I implimented and evaluated Microsoft's GraphRAG due to GraphRAG's popularity in the RAG community as an architecture.

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/success_rates.png" alt="Success Rates" style="width: 70%">
</p>
<p align="center">
  <em>The evaluation success rate of each pipeline.</em>
</p>

<a name="data"></a>
## The Data

The dataset is a memoir written by my grandfather called "My Life Story". The memoir is about 42,000 words which I split into 10 chapters. Each chapter was saved as a seperate PDF file so I could practice and and impliment pipelines that could preprocess PDF documents.

<a name="basic-rag"></a>
## Basic Rag


