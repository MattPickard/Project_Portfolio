# Querying my Grandfather's Memoir with RAG
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

In this project I dive deep into the fascinating world of RAG (Retrieval-Augmented Generation) techniques, applying them to a personal treasure — a memoir written by my grandfather. RAG has revolutionized how we interact with information, powering everything from sophisticated search engines to intelligent customer support systems, personalized content generation, and organizational knowledge management. I crafted and evaluated four distinct RAG pipelines, each designed to showcase different methodologies and highlight the power of an ensemble approach. Starting with a basic RAG implementation as my foundation, I progressively enhanced it with advanced techniques including **strategic reranking**, **context enrichment windows**, and **intelligent query rewriting**. As a final comparison, I implemented Microsoft's celebrated GraphRAG architecture to assess how this cutting-edge approach performs against my custom-built solutions.
<br>
<br>
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/success_rates.png" alt="Success Rates" style="width: 60%">
</p>

<a name="data"></a>
## Data

The dataset is a deeply personal one — my grandfather's memoir titled "My Life Story," a rich narrative spanning approximately 42,000 words. I carefully divided it into 10 chapters, each saved as a separate PDF file. This structure allowed me to practice developing robust pipelines capable of processing and extracting insights from multiple PDF documents.

<a name="basic-rag"></a>
## Basic RAG Implementation 
**Code:** [Basic RAG Pipeline Implementation](https://github.com/MattPickard/Project_Portfolio/blob/main/Memoir_Rag_Project/basic_rag.ipynb)  
**Success Rate:** 60%  


This pipeline, developed using LangChain, serves as both a performance benchmark and the architectural foundation for the more sophisticated ensemble approaches. It implements RAG in its purest form, without additional enhancements. The core workflow involves transforming text chunks into vector embeddings, storing them in a searchable database, and retrieving the most semantically relevant chunks when responding to queries.

To transform the PDFs into usable data, I leveraged PyPDFLoader for initial text extraction. The RecursiveCharacterTextSplitter then divided this content into manageable 1000-character chunks with 200-character overlaps to maintain contextual continuity. I carefully tagged each chunk with its chapter source, hoping the language model would incorporate this into it's response (an expectation that wasn't met, as detailed below). Each text fragment was then embedded using OpenAI's Embeddings API and indexed in a FAISS (Facebook AI Similarity Search) vector database for similarity searches.

When a user poses a question, the system identifies and retrieves the two most contextually relevant chunks from the vector database, which then fuel OpenAI's GPT-4o-mini model to craft a tailored response. The model receives this guidance through the following prompt template:
```
You are querying a memoir called "My Life Story" written by George Shambaugh.
For the question below, provide a concise but sufficient answer. If you don't know, only write "The RAG retrieval
was unable to provide sufficient context":
    {context}
    Question
    {question}
```

### Basic RAG Insights:

Through testing, I uncovered two critical limitations in this initial implementation:

1. The text chunks suffered from contextual fragmentation across page boundaries—a direct consequence of PyPDFLoader's default page-by-page processing approach.
2. Relying exclusively on the top two most similar chunks often failed to provide the model with sufficient context, leaving it unable to generate fully informed responses.

The 60% success rate clearly signals that while basic RAG provides a solid foundation, it falls short of delivering consistently reliable results on its own. This insight drove my exploration of enhanced techniques in the subsequent [RAG Ensemble Pipeline](#rag-ensemble), where I systematically addressed these limitations to dramatically boost performance.

**Additional Note:** I discovered an intriguing behavior:
 OpenAI's GPT-4o-mini model consistently omitted source information from its responses, despite various prompt engineering attempts. This persistence suggests the behavior may be rooted in OpenAI's policy enforcement through RLHF training or system prompts. A future direction would be to compare this behavior across different language models. For practical applications, a simple workaround would be to programmatically append source metadata to responses as a post-processing step.

<a name="rag-ensemble"></a>
## RAG Ensemble Implementation
**Code:** [RAG Ensemble Pipeline Implementation](https://github.com/MattPickard/Project_Portfolio/blob/main/Memoir_Rag_Project/rag_ensemble.ipynb)  
**[Version 1](#version-1) Success Rate:** 85%  
**[Version 2](#version-2) Success Rate:** 95%  

<a name ="version-1"></a>
### Version 1:
My ensemble pipelines demonstrate how to transform a basic RAG implementation into a powerhouse of information retrieval through strategic enhancements—specifically **intelligent reranking** and **context enrichment windows**. I tackled the chunking limitations from the basic pipeline by employing Fitz for superior PDF-to-text conversion and implementing a manual chunking strategy. Rather than simply appending chapter sources to chunks, I preserved this information as metadata—laying groundwork for potential future techniques that could leverage such structured information.

**Cross-Encoder Reranking**  
In this first ensemble iteration, I implemented **cross-encoder 
reranking**, as opposed to LLM-based reranking. Cross-encoder reranking is a 
technique that employs a model (in this case, ms-marco-MiniLM-L-6-v2) to evaluate 
the retrieved chunks against the query and reorders them based on their relevance 
scores. I retrieved the top 10 relevant chunks, applied the reranking model's 
output to reorder them, and then passed the top 3 chunks to the context 
enrichment windows.

**Context Enrichment Windows**  
This technique enhances retrieved chunks by appending the context from the previous and following sections, while also taking into account the overlap. This approach is highly beneficial as it allows for searching smaller chunks, leading to more precise retrieval and reranking. The added surrounding context is then applied for response generation, significantly aiding the model in generating a more informed response.

### Version 1 Insights
With a success rate of 85%, this pipeline represented a notable improvement over 
the [Basic RAG Pipeline](#basic-rag). However, testing revealed a linguistic challenge: the pipeline struggled with queries that used third-person pronouns like "he" or "his" when referring to my grandfather. This difficulty stems from the memoir's first-person perspective creating a semantic disconnect with third-person queries. This insight influenced my implementation of **query rewriting** in version 2.

<a name ="version-2"></a>
### Version 2:

Building on the insights gained from version 1, version 2 of the ensemble pipeline introduced two new techniques: **LLM-based Reranking** (replacing the previous cross-encoder reranking) and **Query Rewriting**, while keeping the **Context Enrichment Windows** technique.

**LLM-based Reranking**  
I explored the capabilities of LLM-based reranking as an alternative to the cross-encoder approach. To do this, I call OpenAI's GPT-4o-mini model 
to rerank the top 10 chunks retrieved using the following prompt:
```
On a scale of 1-10, rate the relevance of the following chunk from George Shambaugh's memoir to the query. Consider the specific
context and intent of the query, not just keyword matches.
    Query: {query}
    Document: {doc}
    Relevance Score:
```
The model's understanding allows for extracting relevance scores that capture more nuanced semantic subtleties. The top 3 highest-scoring chunks are then selected for further processing. While this approach delivers superior performance compared to cross-encoder reranking, it does introduce additional processing time and cost—a trade-off worth considering in production environments.

**Context Enrichment Windows**  
This section remains unchanged, as it utilizes the same technique implemented in version 1.

**Query Rewriting**  
I had two objectives for implementing query rewriting. First, I aimed to convert queries into the first-person perspective to better align with the memoir's writing style, which would enhance retrieval accuracy (this is a problem that is rather unique to this project). Second, I wanted to generate multiple semantically equivalent variations of each query to cast a wider retrieval net while preserving the original intent. After some prompt engineering, I developed the following that effectively achieves both of these objectives:
```
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
The following query is a question pertaining to George Shambaugh's life. Reword the same question in 3 very concise ways, using examples 
of first-person as if George is asking himself and third-person as if someone else is asking about him.

    Original query: {original_query}

    Rewritten query:
```

### Version 2 Insights:

This implementation convincingly demonstrates the power of ensemble approaches in RAG systems. However, it's crucial to recognize that increased sophistication often comes with performance costs. For my specific use case, the additional processing time was acceptable, but other applications might prioritize speed differently. Effective RAG implementation requires thoughtful balancing of complexity against performance requirements.

I also discovered the critical importance of prompt engineering in complex RAG systems. As pipeline sophistication increases, the quality of prompt templates becomes increasingly determinative of both retrieval accuracy and response quality — a factor that shouldn't be underestimated.


<a name ="microsoft-graphrag"></a>
## Microsoft GraphRAG Implementation  
**Code:** [Microsoft GraphRAG Pipeline Implementation](https://github.com/MattPickard/Project_Portfolio/blob/main/Memoir_Rag_Project/microsoft_graphrag.ipynb)  
**Success Rate:** 60%  

I've long been intrigued by GraphRAG's reputation in the RAG community and was eager to evaluate its performance firsthand. My implementation here is deliberately streamlined, similar to the [Basic RAG Pipeline](#basic-rag), without additional enhancements — providing a clean comparison of the core architecture.

GraphRAG takes a fundamentally different approach than traditional RAG pipelines. It constructs a sophisticated knowledge graph where nodes represent text chunks and edges capture the semantic relationships between them. During retrieval, these relationship networks are leveraged to identify the most contextually relevant information for a given query. This architecture excels at understanding connections between concepts distributed throughout a document, making it particularly powerful for handling queries that benefit from synthesizing information across multiple sections.

### GraphRAG Insights:

Testing revealed GraphRAG's impressive ability to generate comprehensive answers to broad, conceptual queries that benefit from connecting information across different parts of the memoir. However, it surprisingly struggled with narrowly focused queries requiring precise information retrieval. Additionally, GraphRAG demands significantly higher computational resources, resulting in noticeably longer processing times. While GraphRAG represents a powerful addition to the RAG ecosystem, my findings suggest it works best as a complementary approach rather than a direct replacement for traditional architectures. An ideal implementation might intelligently route queries to either GraphRAG or traditional pipelines based on query characteristics, or employ one as a fallback when the other fails to produce satisfactory results.

For those interested in implementing GraphRAG, note that many online examples are outdated relative to Microsoft's current library. Feel free to use my implementation as an up-to-date reference.

<a name="evaluation"></a>
## Evaluation
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/evaluation_breakdown.png" alt="Evaluation Breakdown" style="width: 80%">
</p>

To assess performance, I crafted a diverse test set of 20 queries strategically distributed across all 10 chapters of the memoir. Success was measured by comparing responses against established ground truth answers.

### Question 10
Fascinatingly, all four pipelines failed on question 10—the seemingly straightforward query: "Who was his first-born?" This universal failure highlights a particularly challenging case that combines pronoun ambiguity (which query rewriting aimed to address) with a unique content structure challenge. The memoir begins with an extensive family genealogy section that contains numerous extended family relationships, overwhelming the semantic search with competing examples when looking for "first-born." Adding to the complexity, this genealogy section relies heavily on family tree diagrams that aren't semantically searchable in their current format.

### Solving Question 10
A straightforward solution would be to transform the family tree diagrams into searchable text format, making this critical information accessible to the semantic search process.

A more complex solution worth exploring could be to implement a preliminary step into the pipeline that first narrows down the retrieval space by comparing the query to comprehensive summaries of each chapter. For example, when presented with the query "Who was his first-born?", the pipeline would first check which chapters most likely contain information about the birth of his children, then narrow down the search space to the most relevant chapters.

### Evaluation Takeaways
The results reveal interesting performance patterns: traditional RAG architectures consistently outperformed GraphRAG for specific, targeted queries, while GraphRAG demonstrated superior capability in synthesizing broader contextual information—highlighting the complementary strengths of different approaches.

The dramatic performance improvement in ensemble pipelines — jumping from 60% to 95% success — demonstrates how strategic enhancements like reranking and context enrichment can transform RAG effectiveness. However, this improvement comes with increased computational demands and longer processing times, reinforcing the importance of thoughtfully balancing complexity against performance requirements when designing production systems.

## Conclusion
RAG pipelines have emerged as transformative tools in modern natural language processing, enabling unprecedented access to information from both internal and external knowledge sources. Their remarkable flexibility and capacity for enhancement through specialized techniques make them invaluable for building next-generation information retrieval systems. This project demonstrates how thoughtful implementation and strategic enhancements can dramatically improve performance for specialized knowledge domains. I enthusiastically recommend similar explorations to anyone interested in this rapidly evolving field.
