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

# 🚀 Running the MStar Pipeline with Hydra

MStar uses [Hydra](https://hydra.cc/) for flexible, command‑line driven configuration.  
All pipeline settings live in `project-m/src/mstar/config/pipelines/default.yaml`.  
Below you’ll find:

1. **How to launch the app** – `poetry run python -m mstar.main`.
2. **Quick‑start command snippets** – toggle any part of the workflow with Hydra’s _inline overrides_.
3. **A full table of every configuration key** – so you know exactly what each setting does.

---

## 1️⃣ Launching the Pipeline

```bash
# Install dependencies (one‑time)
poetry install

# Run the main entry point
poetry run main
```

The default launch runs _everything_ that is enabled in the `main_runner` section of `default.yaml`.  
If you want to run only a subset, override the booleans on the command line.

---

## 2️⃣ Hydra CLI – Inline Overrides

Hydra lets you flip any flag or tweak any parameter with a single flag on the command line.  
Below are the most common toggles:

| Feature                    | Default | Example Override                             |
| -------------------------- | ------- | -------------------------------------------- |
| Run main indexing pipeline | false   | main_runner.run_main_indexer=false           |
| Run inference              | false   | main_runner.run_inference=true               |
| Build Lightrag index       | false   | main_runner.lightrag_index=true              |
| Run Lightrag inference     | false   | main_runner.lightrag_inference=true          |
| Generate dynamic questions | false   | main_runner.dynamic_question_generation=true |
| Compare answers            | false   | main_runner.answer_comparison=true           |
| Run a single MStar query   | false   | main_runner.single_query_mstar.enable=true   |

### Example Commands

| Purpose                                      | Command                                                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Run only the main indexing steps**         | poetry main mstar.main main_runner.run_main_indexer=true                                                                              |
| **Skip indexing, only inference**            | poetry main mstar.main main_runner.run_main_indexer=false main_runner.run_inference=true                                              |
| **Build Lightrag index, then run inference** | poetry main mstar.main main_runner.lightrag_index=true main_runner.lightrag_inference=true                                            |
| **Generate a single query answer**           | poetry main mstar.main main_runner.single_query_mstar.enable=true main_runner.single_query_mstar.query="TI net profit for q4 of 2024" |
| **Dynamic question generation**              | poetry main mstar.main main_runner.dynamic_question_generation=true                                                                   |
| **Compare two answer modes**                 | poetry main mstar.main main_runner.answer_comparison=true                                                                             |

> **Tip:** Hydra automatically resolves `${...}` placeholders, so you don’t need to change the paths manually unless you’re moving the project.

---

## 3️⃣ Full Configuration Table

Below is a flattened view of every key in `default.yaml`.  
Dot‑notation (`project.name`) is used for nested keys.

| Key                                                                                  | Value                                                                                                                                                              |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| project.LLM.deep_model_name                                                          | deepcoder:14b                                                                                                                                                      |
| project.LLM.model_tools                                                              | granite3.3:8b                                                                                                                                                      |
| project.LLM.embedder_name                                                            | nomic-embed-text                                                                                                                                                   |
| project.LLM.reason_model                                                             | phi4-reasoning:14b-plus-q4_K_M                                                                                                                                     |
| project.LLM.embedder_dim                                                             | 768                                                                                                                                                                |
| project.LLM.grok_api                                                                 | abc                                                                                                                                                                |
| project.LLM.openai_api_key                                                           | abc                                                                                                                                                                |
| project.name                                                                         | TI                                                                                                                                                                 |
| project.project_path                                                                 | /Desktop/repos/project-m                                                                                                                                           |
| project.data_path                                                                    | ${project.project_path}/eval/temp_ti_data                                                                                                                          |
| project.base_cache_path                                                              | ${project.project_path}/eval/temp_ti_cache                                                                                                                         |
| project.base_prompt_path                                                             | ${project.project_path}/src/mstar/prompts                                                                                                                          |
| project.domain                                                                       | Financial Reporting                                                                                                                                                |
| project.pipelines.extract_pdf.pdf_dir                                                | ${project.data_path}/user_data_input_files                                                                                                                         |
| project.pipelines.extract_pdf.stage_dir                                              | ${project.base_cache_path}/extract_pdf                                                                                                                             |
| project.pipelines.extract_pdf.cache_file_name                                        | extract_pdf_output                                                                                                                                                 |
| project.pipelines.split_chunk_files.chunk_size                                       | 500                                                                                                                                                                |
| project.pipelines.split_chunk_files.cache_chunks_dir                                 | ${project.base_cache_path}/cache_chunks                                                                                                                            |
| project.pipelines.split_chunk_files.chunk_pipeline_output_name                       | chunk_pipeline_output                                                                                                                                              |
| project.pipelines.process_chunked_tables_pipeline.cache_process_chunked_tables_dir   | ${project.base_cache_path}/cache_process_chunked_tables                                                                                                            |
| project.pipelines.process_chunked_tables_pipeline.process_chunked_tables_output_name | processed_tables_output                                                                                                                                            |
| project.pipelines.chunked_summerizer_pipeline.cache_chunked_summerizer_dir           | ${project.base_cache_path}/chunked_summerizer                                                                                                                      |
| project.pipelines.chunked_summerizer_pipeline.chunked_summerizer_output_name         | chunked_summerizer_output                                                                                                                                          |
| project.pipelines.chunked_summerizer_pipeline.prompt_template_path                   | ${project.base_prompt_path}/summarization.txt                                                                                                                      |
| project.pipelines.chunked_summerizer_pipeline.prompt_inputs                          | - format_instructions<br>- content                                                                                                                                 |
| project.pipelines.NER.cache_ner_dir                                                  | ${project.base_cache_path}/cache_ner                                                                                                                               |
| project.pipelines.NER.ner_output_name                                                | cache_ner_output                                                                                                                                                   |
| project.pipelines.NER.faiss_summary_retriever_k                                      | 5                                                                                                                                                                  |
| project.pipelines.NER.faiss_md_retriever_k                                           | 5                                                                                                                                                                  |
| project.pipelines.NER.prompt_template_path                                           | ${project.base_prompt_path}/ner.txt                                                                                                                                |
| project.pipelines.NER.prompt_inputs                                                  | - format_instructions<br>- support_docs<br>- real_data                                                                                                             |
| project.pipelines.resolve_unknown_entity.cache_rue_dir                               | ${project.base_cache_path}/rue_cache                                                                                                                               |
| project.pipelines.resolve_unknown_entity.rue_output_name                             | rue_output                                                                                                                                                         |
| project.pipelines.resolve_unknown_entity.prompt_template_path                        | ${project.base_prompt_path}/resolve_unknown_entity.txt                                                                                                             |
| project.pipelines.resolve_unknown_entity.prompt_inputs                               | - name<br>- relation_description<br>- real_data<br>- format_instructions                                                                                           |
| project.pipelines.entity_graph.cache_entity_graph_npz_dir                            | ${project.base_cache_path}/entity_graph_cache                                                                                                                      |
| project.pipelines.entity_graph.cache_entity_graph_faissindex_dir                     | ${project.base_cache_path}/faiss_index_cache                                                                                                                       |
| project.pipelines.entity_graph.ent_graph_cache_attr                                  | - entities<br>- relations<br>- id2pos<br>- pos2id<br>- next_pos<br>- ent_type_label2id<br>- rel_type_label2id<br>- emb_store<br>- params<br>- dim<br>- id2metadata |
| project.pipelines.query_answer.prompt_template_path                                  | ${project.base_prompt_path}/query_ner.txt                                                                                                                          |
| project.pipelines.query_answer.eval_q_path                                           | ${project.project_path}/eval                                                                                                                                       |
| project.pipelines.query_answer.eval_file_name                                        | openai_ans_result_tuned_last.csv                                                                                                                                   |
| project.pipelines.query_answer.eval_result_file_name                                 | naive_rag_answers.csv                                                                                                                                              |
| project.pipelines.query_answer.eval_ner_result_file_name                             | ner_result_answers.csv                                                                                                                                             |
| project.pipelines.query_answer.ti_question_path                                      | ${project.data_path}/TI_edited_questions.txt                                                                                                                       |
| project.pipelines.query_answer.cache_answer_dir                                      | ${project.base_cache_path}/entity_graph_cache                                                                                                                      |
| project.pipelines.query_answer.cache_answer_name                                     | query_cache_result_150q                                                                                                                                            |
| project.main_runner.run_main_indexer                                                 | true                                                                                                                                                               |
| project.main_runner.run_inference                                                    | false                                                                                                                                                              |
| project.main_runner.lightrag_index                                                   | false                                                                                                                                                              |
| project.main_runner.lightrag_inference                                               | false                                                                                                                                                              |
| project.main_runner.light_rag_workingdir                                             | ${project.base_cache_path}/lightrag_wd                                                                                                                             |
| project.main_runner.lightrag_inference_mode                                          | mix                                                                                                                                                                |
| project.main_runner.dynamic_question_generation                                      | false                                                                                                                                                              |
| project.main_runner.dynamic_question_template_path                                   | ${project.base_prompt_path}/question_generation.txt                                                                                                                |
| project.main_runner.answer_comparison                                                | false                                                                                                                                                              |
| project.main_runner.answer_1_mode                                                    | mix                                                                                                                                                                |
| project.main_runner.answer_2_mode                                                    | mstar_final                                                                                                                                                        |
| project.main_runner.single_query_mstar.enable                                        | false                                                                                                                                                              |
| project.main_runner.single_query_mstar.query                                         | TI net profit for q4 and q3 of year 2024                                                                                                                           |
| version_base                                                                         | None                                                                                                                                                               |

> **NOTE** – Any `${...}` reference will be resolved by Hydra automatically.  
> If you move the repository, just update `project.project_path` and the rest will follow.

---

### 🎉 Get Started

```bash
# Install Poetry if you haven’t already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repo and enter it
git clone https://github.com/yourorg/project-m.git
cd project-m

# Install dependencies
poetry install

# Run the default pipeline
poetry run python main
```

### Pipelines Explanation

- run_main_indexer: Equal to the indexer pipeline of the paper, to convert pdf or md to concept-type graph
- single_query_mstar: Running one query againes indexed data.
- dynamic_question_generation: To generate the questions necessary for the evaluation phase.
- run_inference: Running inference against full questions, pipeline two.
- lightrag_index, lightrag_inference: Running indexing and question answering using LightRAG
- answer_comparison: Final evaluation of the three method, mstar, lightrag and naiverag

### IMPORTANT NOTE

- First you need to be sure ollama is running on your machine with models, granite3.3 and nomic-text-embedding. Without these two code will not work.
- In order to reproduce the result you need the main data directory, `eval` which is removed from the repo. To access the data contact the author of the paper on the [Linkedin](https://linkedin.com/in/omidnw).
- Happy hacking! 🚀
