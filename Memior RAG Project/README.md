# Querying my Grandfather's Memoir with RAG Pipelines
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/family_photo.jpg" alt="Family Photo" style="width: 60%;">
</p>
<p align="center">
<em>My grandfather is holding me on the right.</em>
</p>

## Table of Contents  
<table>
  <tr>
    <td><a href="#introduction">Introduction</a><br>
    <a href="#data">Data</a><br>
    <a href="#basic-rag">Basic RAG</a><br>
    <a href="#rag-ensemble">RAG Ensemble</a><br>
    <a href="#microsoft-graphrag">Microsoft GraphRAG</a><br>
    <a href="#evaluation">Evaluation</a><br>
    <a href="#conclusion">Conclusion</a></td>
  </tr>
</table>

## Implementation Resources
* [https://github.com/NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
* [https://github.com/microsoft/GraphRAG](https://github.com/microsoft/GraphRAG)

<a name="introduction"></a>
## Introduction

This project serves as an demonstration and exploration of various RAG (Retrieval-Augmented Generation) techniques and apply them to a memoir written by my grandfather for evaluation. RAG is used in a wide range of applications, including enhancing search engines, improving customer support systems and chatbots, generating personalized content, and facilitating knowledge management within organizations. In this project, I built and evaluated four distinct RAG pipelines, each designed to compare and contrast methodologies and architectures and to showcase the importance and power of utilizing an ensemble approach. I began by building a basic RAG pipeline to serve as a baseline, and subsequently added on an ensemble of techniques on top of it, including **reranking**, **context enrichment windows**, and **query rewriting**. Finally, I built and evaluated a simple implementation of Microsoft's GraphRAG, a popular architecture in the RAG community, to assess its baseline performance and capabilities.

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/success_rates.png" alt="Success Rates" style="width: 70%">
</p>
<p align="center">
  <em>The evaluation success rate of each pipeline.</em>
</p>

<a name="data"></a>
## Data

The dataset consists of my grandfather's memoir titled "My Life Story," consisting of approximately 42,000 words, of which I divided into 10 chapters. Each chapter was saved as a separate PDF file, allowing me to practice implementing pipelines that preprocess PDF documents.

<a name="basic-rag"></a>
## Basic RAG Implementation 
**Code:** [Basic RAG Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/basic_rag.ipynb)  
**Success Rate:** 60%  


This pipeline, developed using LangChain, serves as a baseline for evaluation and a foundation for the ensemble pipelines. It implements a basic RAG architecture without incorporating any additional techniques. The core functionality involves storing text chunks into a vector database through encoding, followed by querying the database to retrieve the two most similar chunks for response generation. 

To preprocess the text, I utilized PyPDFLoader to convert the PDF files into plain text. I employed RecursiveCharacterTextSplitter to divide the text into chunks of 1000 characters, allowing for 200-character overlaps between chunks. I added the chapter source at the end of each chunk, hoping that the language model would utilize this context in its responses (however, this did not occur, as noted in the "Additional Note" section below). Each chunk was then embedded using the OpenAI Embeddings API and stored in a vector database created with FAISS (Facebook AI Similarity Search).

When a query is made, the database retrieves the top two most similar chunks from the vector database, which are then passed to OpenAI's GPT-4o-mini model to generate a response. The following template was used to prompt the model with the context and query:
```
You are querying a memoir called "My Life Story" written by George Shambaugh.
For the question below, provide a concise but sufficient answer. If you don't know, only write "The RAG retrieval
was unable to provide sufficient context":
    {context}
    Question
    {question}
```

### Basic RAG Insights:

Two significant weaknesses were identified while testing this pipeline:

1. The text chunks do not maintain context across pages. This issue arises from PyPDFLoader's default behavior, which splits documents into separate pages.
2. Relying on the top two most similar chunks from the vector database often results in insufficient information being provided to the model.

A success rate of 60% indicates that while a basic RAG pipeline is insufficient for achieving high success rates on its own, but its performance can be improved by incorporating additional techniques. In the subsequent [RAG Ensemble Pipeline](#rag-ensemble) implementation, I apply various techniques that effectively address the weaknesses above and enhance the pipeline's performance.

**Additional Note:** I noticed that OpenAI's GPT-4o-mini model consistently failed to append source information to the end of its responses. This issue persisted despite modifying the prompts and repositioning the sources within the context. It is possible that this behavior is due to the model conforming to an OpenAI policy reinforced during RLHF training or VIA instructions given to it in its system prompt. If I were to continue this project, an interesting direction to explore would be to see if other models are more flexible. A manual solution to overcome this would be to implement a step in the pipeline that appends source metadata to the end of the response.

<a name="rag-ensemble"></a>
## RAG Ensemble Implementation
**Code:** [RAG Ensemble Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/rag_ensemble.ipynb)  
**[Version 1](#version-1) Success Rate:** 85%  
**[Version 2](#version-2) Success Rate:** 95%  

<a name ="version-1"></a>
### Version 1:
The ensemble pipelines demonstrate how to improve upon a basic RAG pipeline using additional techniques, specifically **reranking** and **context enrichment windows**. I also addressed the chunking issue from the previous Basic RAG Pipeline by using Fitz to convert PDF files to text and manually splitting the text into chunks. Instead of appending the chapter source to each chunk, I saved it as metadata. While this pipeline doesn't make use of that metadata, there are many techniques available that can leverage such metadata to improve retrieval and response quality.

**Cross-Encoder Reranking**  
In this first version of the ensemble pipeline, I implemented **cross-encoder reranking**, as opposed to LLM-based reranking. Cross-encoder reranking is a technique that employs a model (in this case, ms-marco-MiniLM-L-6-v2) to evaluate the retrieved chunks against the query and reorders them based on their relevance scores. I retrieved the top 10 relevant chunks, applied the reranking model's output to reorder them, and then passed the top 3 chunks to the context enrichment windows.

**Context Enrichment Windows**  
This technique enhances retrieved chunks by appending the context from the previous and following sections, while also taking into account the overlap. This approach is highly beneficial as it allows for searching smaller chunks, leading to more precise retrieval and reranking. The added surrounding context is then applied for response generation, significantly aiding the model in generating a more informed response.

### Version 1 Insights

With a success rate of 85%, this pipeline represented a notable improvement over the [Basic RAG Pipeline](#basic-rag). However, during testing, I discovered that it struggled with queries that exclusively used pronouns such as "he" or "his" when referring to my grandfather. I believe this issue arises because the memoir is written in the first-person perspective, while queries phrased in the third-person leads to a slight semantic mismatch. To address this challenge, I introduced a **query rewriting** step in version 2 of the ensemble pipeline.

<a name ="version-2"></a>
### Version 2:

Building on the insights gained from version 1, version 2 of the ensemble pipeline introduced two new techniques: **LLM-based Reranking** (replacing the previous cross-encoder reranking) and **Query Rewriting**, while keeping the **Context Enrichment Windows** technique.

**LLM-based Reranking**  
I explored the performance of an LLM-based reranking model after previously using the cross-encoder reranking model. To do this, I call OpenAI's GPT-4o-mini model to rerank the top 10 chunks retrieved using the following prompt:
```
On a scale of 1-10, rate the relevance of the following chunk from George Shambaugh's memoir to the query. Consider the specific
context and intent of the query, not just keyword matches.
    Query: {query}
    Document: {doc}
    Relevance Score:
```
The score from the model's response is extracted, and the top 3 chunks are selected based on their scores to be passed on for further processing. I find this approach to be more intuitive and it has shown better performance compared to the cross-encoder reranking model; however, it does come with increased processing time and cost.

**Context Enrichment Windows**  
This section remains unchanged, as it utilizes the same technique implemented in version 1.

**Query Rewriting**  
I had two objectives for implementing query rewriting. First, I aimed to convert queries into the first-person perspective to better align with the memoir's writing style, which would enhance retrieval accuracy (this is a problem that is rather unique to this project). Second, I wanted to create multiple variations of the same query while preserving its meaning to improve the retrieval results. After some prompt engineering, I developed the following that effectively achieves both of these objectives:
```
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
The following query is a question pertaining to George Shambaugh's life. Reword the same question in 3 very concise ways, using examples 
of first-person as if George is asking himself and third-person as if someone else is asking about him.

    Original query: {original_query}

    Rewritten query:
```

### Version 2 Insights:

I feel confident that this implementation demonstrates the effectiveness of an ensemble approach to RAG. However, it is crucial to understand that incorporating more techniques and complexity into ensemble designs can lead to slower pipeline performance. In my specific use case, the wait times were acceptable; however, other use cases may prioritize speed. Any RAG implementation should recognize and balance the trade-off between complexity and speed.

Additionally, I want to highlight the critical role of prompt engineering in the success of this pipeline, which becomes more vital as the complexity of the system increases. The effectiveness of both retrieval and response quality is significantly influenced by the prompt templates utilized.


<a name ="microsoft-graphrag"></a>
## Microsoft GraphRAG Implementation  
**Code:** [Microsoft GraphRAG Pipeline Implementation](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Memior%20Project/microsoft_graphrag.ipynb)  
**Success Rate:** 60%  

I have always been interested in evaluating the performance of GraphRAG due to its popularity in the RAG community. This was a simple implementation, resembling the [Basic RAG Pipeline](#basic-rag), and did not incorporate any additional techniques.

GraphRAG's architecture differs from the previous pipelines and is somewhat more complex. It constructs a graph where nodes represent chunks of text, and the edges between these nodes indicate the similarity between the chunks. During the retrieval phase, the relationships represented by the edges are utilized to identify the most relevant chunks for a given query. This method allows GraphRAG to effectively "understand" relationships between context chunks that are spread throughout the documents, making it particularly effective for handling broad queries.

### GraphRAGInsights:

During testing, I was impressed with GraphRAG's ability to generate comprehensive answers to queries that benefit from broader contexts. However, it struggled with narrow queries that required retrieving very specific information. Additionally, GraphRAG incurs a significant higher computational cost, resulting in longer wait times. While GraphRAG is a powerful tool, and has its place in the RAG community, I am not convinced that it should serve as a direct replacement for traditional RAG architecture due its own limitations on narrow queries. It may make more sense to use it in tandem with traditional pipelines. For instance, one could be triggered as a fallback when the other pipeline fails to make an adiquate response or the pipeline can determine which approach to use depending on the query.

One thing to note about Microsoft GraphRAG is that many online implementations I found were out-of-date with the latest Microsoft GraphRAG library, so feel free to use this example as an up-to-date reference. 

<a name="evaluation"></a>
## Evaluation
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/evaluation_breakdown.png" alt="Evaluation Breakdown" style="width: 80%">
</p>

For evaluation, I created a test set of 20 queries spread evenly across the 10 chapters of the memoir. A correct answer was a response that matched the ground truth answer. 

### Question 10
Interestingly, all four pipelines failed to answer question 10, which was a seemingly simple query: "Who was his first-born?" This query is an example a user prompting with an unspecific pronoun, an issue I aimed to address through query rewriting. However, this particular query posed a unique challenge because of the content and layout of section one of the memoir. The first section of the memoir is a detailed family genealogy which contains and breaks down family relationships. This caused the retrieval process to be flooded with the many examples of family relationships in the genealogy section, overwhelming the retrieval process with various examples of family relationships when it semantically searched for "first-born". To add and extra layer of difficulty, the geneology section's formatting relies heavily on diagrams of family trees, which are not semantically seachable.

### Solving Question 10
A straightforward solution to this problem would be to implement a process that transforms the family tree diagrams into text format. This text would be semantically searchable and comprehensible to the model, allowing for more effective retrieval of information.

A more complex solution worth exploring could be to implement a step into the pipeline that first narrows down the retrieval space by comparing the query to comprehensive summaries of each chapter. For example, when presented with the query "Who was his first-born?", the pipeline would first check which chapters most likely contain information about the birth of his children, then narrow down the search space to the most relevant chapters.

### Evaluation Takeaways
Overall, the classic RAG architectures demonstrated greater success for specific queries compared to the GraphRAG architecture. However, GraphRAG excelled in retrieving broader context, showcasing its unique strengths. This can be seen from the varying performance of the two basic pipelines on different queries.

The ensemble pipelines illustrate how incorporating additional techniques, such as reranking and context enrichment windows, can significantly enhance the success rate of retrieval tasks. Nevertheless, this improvement often comes at the expense of longer wait times and increased computational costs. Therefore, it is essential to balance the trade-off between complexity and speed when designing RAG pipelines. 

## Conclusion
RAG pipelines are essential in many modern-day natural language processing applications, allowing for the quick retrieval of relevant information from internal and external datasets. Their flexibility and capacity for improvement through various techniques make them an excellent option for boosting information retrieval systems. I highly recommend similar projects to anyone interested in exploring this field. If you have any questions or comments, please feel free to reach out.
