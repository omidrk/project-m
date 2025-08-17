# Introduction

**Retrieval-Augmented Generation (RAG)** [^1] is one of the most widely used methods in today’s AI landscape. It enhances large language models (LLMs) by combining the retrieval of relevant external documents with text generation. By accessing a large corpus of text data, RAG fetches the most similar documents based on vector similarity and feeds them to the model as context. This enables AI applications to adapt to new, unseen domains without requiring fine-tuning.  

While RAG allows for real-time data retrieval and rapid domain adaptation, it still lacks deep semantic understanding between text chunks and struggles with contextual coherence across multiple documents.

---

## Graph-based Extensions to RAG

Graph-based methods have been incorporated to address RAG’s shortcomings. Graphs are particularly effective at capturing relationships, enabling sense-making across large text corpora.  

- **GraphRAG** decomposes text into entities and relations, forming community-based summaries at different levels. This hierarchical approach helps the model answer queries from top-down or bottom-up perspectives, before aggregating into a global answer. While GraphRAG improves contextual understanding, it often includes irrelevant information and incurs heavy processing costs.  
- **LightRAG** improves entity indexing and retrieval efficiency by combining detailed and conceptual retrieval. This leads to more relevant, higher-quality answers. However, retrieval still suffers from redundancy, resulting in noise and degraded model performance.

Despite these advances, several challenges remain for enterprise applications:

- **Dependency on knowledge graphs (KGs):** Most enterprise data is structured but not fully available in a unified form necessary for KG construction.  
- **Data privacy and governance:** Existing approaches assume full data availability and do not allow sensitive information to remain at the source.  
- **Large model dependency:** Handling large contexts often requires massive proprietary LLMs, unsuitable for privacy-sensitive use cases.  
- **Query decomposition:** Current methods leave decomposition to the LLM, which often produces partial answers when retrieved data is incomplete. Complex, multi-hop queries remain especially challenging.

---

## Our Approach

To overcome these limitations, we introduce a **novel graph-based retrieval framework** leveraging **iterative concept-type path retrieval**.  

Instead of relying on a traditional knowledge graph, our method builds a **Concept-Type Graph (CTG)**:  

- **Concepts** are schemas of structured data described by an LLM from small samples.  
- **Types** are domain-specific categories extracted during the entity extraction phase.  

Unlike KGs, a CTG can be constructed from minimal data—such as table rows, columns from PDFs, or text samples—without requiring the full dataset. This approach enhances privacy and data governance.  

### Two Pipelines

1. **Indexing Pipeline**  
   - Efficient entity and relation extraction.  
   - Builds the CTG with reduced data volume and lower GPU memory usage.  

2. **Querying Pipeline**  
   - Novel **query decomposition algorithm** identifies all elements of a query with rewriting query in disjunctive-conjuctive form.  
   - Uses the CTG to construct a search space.  
   - Applies **Boolean query rewriting** (disjunction of conjunctions) to prune redundant answer sets.  
   - Narrows the search space before interacting with actual data values.  

This reduces redundant information in the LLM’s context, enabling the use of **lightweight open-source models (~8B parameters)** while achieving performance on par with flagship models.

---

## Contributions

Our work makes the following contributions:

- Analysis of current RAG and graph-based retrieval approaches, highlighting their limitations.  
- Introduction of a new entity extraction approach to capture **types** and **concepts**, enabling **concept graph formation** instead of traditional KGs.  
- Development of a **disjunctive-conjunctive query decomposition** method combined with CTG-based path retrieval.  
- An **iterative path-based query answer evaluation algorithm**.  
- Empirical demonstration that our solution outperforms naive RAG, GraphRAG, and PathRAG.  

---


