# RAG Techniques to Query my Grandfather's Memoir
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/family_photo.jpg" alt="Family Photo" style="width: 60%;">
</p>
<p align="center">
<em>My grandfather is holding me on the right.</em>
</p>

## Table of Contents  
* [Purpose](#purpose)
* [Data](#data)
* [Basic RAG](#basic-rag)
* [RAG Ensemble](#rag-ensemble)
* [Microsoft GraphRAG](#microsoft-graphrag)
* [Evaluation](#evaluation)

## Implementation Resources
* [https://github.com/NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
* [https://github.com/microsoft/GraphRAG](https://github.com/microsoft/GraphRAG)

<a name="purpose"></a>
## Purpose

This project was an exploration of various RAG (Retrieval-Augmented Generation) techniques applied to a dataset of personal significance - a memoir written by my grandfather. The hope was to demonstrate and improve my intuitions and understandings of RAG and NLP (Natural Language Processing) pipelines. In total, I built and evaluated four pipelines with the goal of comparing various techniques and architectures. I started with a simple RAG pipeline to use as a baseline, then worked on an ensemble of techniques including **reranking**, **context enrichment windows**, and **query rewriting**. Finally, I implemented and evaluated Microsoft's GraphRAG due to GraphRAG's popularity in the RAG community as an architecture.

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/success_rates.png" alt="Success Rates" style="width: 70%">
</p>
<p align="center">
  <em>The evaluation success rate of each pipeline.</em>
</p>

<a name="data"></a>
## Data

The dataset is a memoir written by my grandfather called "My Life Story". The memoir is about 42,000 words which I split into 10 chapters. Each chapter was saved as a separate PDF file so I could practice implementing pipelines that preprocess PDF documents.

<a name="basic-rag"></a>
## Basic RAG Implementation 
**Code:** [Basic RAG Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/basic_rag.ipynb)  
**Success Rate:** 60%  


This first pipeline, built with LangChain, is an implementation of a simple RAG architecture without any additional techniques. At its core, it consists of storing chunks the text into a vector database via encodings and then querying the database, retrieving the two most similar chunks to craft a response. To preprocess the text, I used PyPDFLoader to convert the PDF files into text, then used RecursiveCharacterTextSplitter to split the text into 1000 character chunks with 200 character overlaps. I appended the chapter source to the end of each chunk with hopes the language model would later pass it from the context onto its answers (This ended up not being the case, read the "Attitonal Note" below). Each chunk was then embedded using the OpenAI Embeddings API and converted into a vector database using FAISS (Facebook AI Similarity Search).
When queried, the database retrieves the top two most similar chunks and then uses OpenAI's GPT-4o-mini model to generate a response. The following prompt template was provided to the model:
```
You are querying a memoir called "My Life Story" written by George Shambaugh.
For the question below, provide a concise but sufficient answer. If you don't know, only write "The RAG retrieval
was unable to provide sufficient context":
    {context}
    Question
    {question}
```

### Basic RAG Insights:

During the test phase of this pipeline, two major weaknesses were identified:

1. The chunks do not carry over context across pages, which was a result of PyPDFLoader's default behavior of splitting the documents into pages.
2. Relying solely on the top two most similar chunks without any additional techniques often leads to insufficient context retrieval.

With a success rate of 60%, it indicates that a basic RAG pipeline is a good starting point, but is not enough to achieve high success rates by itself. These weaknesses can be addressed by adding on more complexity and techniques. In the following [RAG Ensemble Pipeline](#rag-ensemble) implementation you can see various techniques that effectively improve performance.

**As additional note:** I observed that OpenAI's GPT-4o-mini model consistently failed to append sources to the end of responses despite it being prompted to do so and the sources being present in the provided context. This behavior persisted even after modifying the prompts and repositioning the sources within the context. It is possible that this is the model conforming to an OpenAI policy. If I were to continue this project, I would be curious to see if other models are more accepting of such requests. To overcome this, one option would be to have a separate step that appends source metadata to the end of the response.

<a name="rag-ensemble"></a>
## RAG Ensemble Implementation
**Code:** [RAG Ensemble Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/rag_ensemble.ipynb)  
**[Version 1](#version-1) Success Rate:** 85%  
**[Version 2](#version-2) Success Rate:** 95%  

<a name ="version-1"></a>
### Version 1:
The goal of creating an ensemble pipeline was to improve the success rate of the [Basic RAG Pipeline](#basic-rag) by leveraging additional techniques. I settled on two techniques: **reranking** and **context enrichment windows**. I also fixed the chunking issue caused by PyPDFLoader in the Basic RAG Pipeline. This was handled by using Fitz to convert the PDF files into text, then manually splitting the text into chunks. The chapter source was then saved as metadata instead of being appended to the end of each chunk. This pipeline doesn't end up using that metadata, but there are various techniques out there that can leverage such metadata to help improve retrieval and response quality. 

**Cross-Encoder Reranking**  
This first ensemble version leveraged **cross-encoder reranking**, specifically because I was curious about how it would perform. Cross-encoder reranking is a technique that uses a model (in this case, ms-marco-MiniLM-L-6-v2) to compare the retrieved chunks to the query and reorder the chunks based on their relevance scores. I called the top 10 relevant chunks and then used the reranking model to reorder them and passed the top 3 to the context windows enriched.

**Context Enrichment Windows**  
This technique takes a chunk and, taking into account the chunk overlap, appends the context of the previous and next chunk to the chunk for better context. This is incredibly useful and improves performance by allowing you to search smaller chunks which allows for more precise retrieval and reranking, then adding the surrounding context for the model to use while generating a response.

### Version 1 Insights

At a success rate of 85%, this pipeline was a significant improvement over the [Basic RAG Pipeline](#basic-rag). However, during testing, I found that it struggled with queries that used pronouns such as "he" or "his" when referring to my grandfather. My theory was that this was because the memoir was written from a first-person perspective, and such queries cause a semantic similarity mismatch between the query and the context retrieved. To address this, I added a query rewriting step to the pipeline in version 2.

<a name ="version-2"></a>
### Version 2:

Using what I learned from version 1, version 2 of the ensemble pipeline applied the following techniques: **LLM-based Reranking** (as opposed to cross-encoder reranking), **Context Enrichment Windows**, and **Query Rewriting**.

**LLM-based Reranking**  
After using the cross-encoder reranking model, I wanted to see how an LLM-based reranking model would perform. I reranked the top 10 chunks retrieved by using this prompt:
```
On a scale of 1-10, rate the relevance of the following chunk from George Shambaugh's memoir to the query. Consider the specific context and intent of the query, not just keyword matches.
    Query: {query}
    Document: {doc}
    Relevance Score:
```
The score from the model's response is extracted and the top 3 chunks based on their scores to be passed on. To me, this approach is more intuitive and performed better than the cross-encoder reranking model, but it is also slower and more expensive. 

**Context Enrichment Windows**  
Nothing new here, this is the same technique used in version 1.

**Query Rewriting**  
There were two reasons I wanted to implement query rewriting. First, I wanted to try changing queries to first-person perspective to match the memoir's writing style more accurately for more accurate retrieval. Second, if the queries were rewritten into various versions of the same query while retaining the same meaning, it would aid in semantic retrieval.
After a bit of prompt engineering, I landed on the following prompt which seemed to accomplish both of these goals:
```
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
The following query is a question pertaining to George Shambaugh's life. Reword the same question in 3 very concise ways, using examples of first-person as if George is asking himself and third-person as if someone else is asking about him.

    Original query: {original_query}

    Rewritten query:
```

### Version 2 Insights:

While there are many more techniques that could improve this pipeline's performance, I feel happy that I was able to demonstrate the power of an ensemble approach to RAG. With most ensemble designs, it is important to note that by adding more techniques and complexity, the pipeline becomes slower. Personally, the wait times didn't bother me much for my use case, but there are always considerations with regards to other use cases where speed is a priority. There is a clear complexity-speed trade-off in RAG implimentations that should be considered and balanced.

One more thing I'd like to emphasize is the importance of prompt engineering in the success of this pipeline, and likely even more important in more complex pipelines. The quality of both retrieval and response were heavily dependent on the prompt templates used. 


<a name ="microsoft-graphrag"></a>
## Microsoft GraphRAG Implementation  
**Code:** [Microsoft GraphRAG Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/microsoft_graphrag.ipynb)  
**Success Rate:** 60%  

When starting this project, I was very curious to see how GraphRAG would perform because of its popularity within the RAG community. This was a simple implementation of the architecture, similar to the [Basic RAG Pipeline](#basic-rag) and leveraged no additional techniques.

The underlying architecture of GraphRAG is different from the previous pipelines and a bit more complex. A graph is created from nodes that represent chunks of text, and the edges between nodes represent the similarity between those chunks. During the retrieval phase, the relationships between node edges are used to retrieve the most relevant chunks for a given query. Using this approach, GraphRAG can retrieve and put together context that is spread out throughout the documents, which makes it especially powerful for broad queries.

### GraphRAGInsights:

While testing, I was impressed with GraphRAG's ability to form a "big picture" answer to queries that benefit from broader contexts. However, it became apparent that it really struggles with narrow queries that required retrieving very specific context. It also has a higher computational cost, leading to longer wait times. My intuition tells me that instead of using GraphRAG as a replacement for other architectures, it might be better used to augment pipelines, perhaps as a fallback and or for adding context for broader queries. If you did this, you would need to accept the longer wait times and higher computational costs.

One thing to note is that many online GraphRAG implimentations are out of date with the latest Microsoft GraphRAG library, feel free to use this example as reference. 

<a name="evaluation"></a>
## Evaluation
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/evaluation_breakdown.png" alt="Evaluation Breakdown" style="width: 80%">
</p>

For evaluation, I created a test set of 20 queries spread evenly across the 10 chapters of the memoir. A correct answer was defined as a response that matched the ground truth answer. 

### Question 10
Interestingly, all 4 pipelines failed question 10, which was a rather simple query: "Who was his first-born?" While this was an example of a query that used an unspecific pronoun, an issue that I attempted to address with query rewriting, even more detrimental to the retrieval of this query was the memoir itself. The first section of the memoir is a an in-depth family genealogy, and due to so many examples of family relationships in this section, the retrieval processes became confused by the many semantically similar chunks.  

## Big Takeaways
In general, the classic RAG architecture was more successful than the GraphRAG architecture for specific queries, however, GraphRAG's ability to retrieve broader context ended up being very impressive. To me it seems that adding more techniques such as reranking and context enrichment windows can derastically improve the success rate, with a cost of longer wait times and higher computational costs. Complexity and speed seems to be a big trade-off in RAG pipelines.  
    
### Thank you for reading! 
I had a good time with this project. If you have any questions or comments, feel free to reach out.
