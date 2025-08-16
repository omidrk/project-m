import numpy as np

from os import path, listdir
from mstar.core.logger import setup_logging
import logging
from omegaconf import DictConfig
from typing import Any

from mstar.core.cache import get_client

import re
import os
import json
import time
import asyncio
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_model_complete, ollama_embed


setup_logging()
logger = logging.getLogger(__name__)
client = get_client()


def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    data = data.replace("+", "-")

    # Updated regex to capture questions more reliably
    queries = re.findall(r" - Question\d+: (.+)", data)
    return queries


def extract_queries_ti(file_path):

    with open(file_path, "r") as f:
        data = f.readlines()
    return [d.strip() for d in data]


async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def insert_text(rag: Any, unique_contexts: list):

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


async def initialize_rag():

    # Initialize LightRAG with Ollama model
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
        llm_model_name="granite3.3",  # Your model name
        llm_model_kwargs={"options": {"num_ctx": 32768}},
        embedding_batch_num=1,
        # Use Ollama embedding function
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=2000,  ## nomic embed limitation
            func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text"),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def lightrag_index(cfg: DictConfig):

    logger.info("Configuration loaded.")

    npy_dir = f"{cfg.project.base_cache_path}/extract_pdf"
    npy_files = [
        f for f in listdir(npy_dir) if f.endswith(".npy") and f.startswith("stage_1")
    ]

    npy_data = {}
    for file in npy_files:
        file_path = path.join(npy_dir, file)

        try:
            data = np.load(file_path, allow_pickle=True).item()
            npy_data[data["document"]["filename"]] = data["document"]["md_content"]
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    print(f"Finished loading {len(npy_data)} npy files.")
    unique_contexts = [v for _, v in npy_data.items()]

    WORKING_DIR = cfg.project.main_runner.light_rag_workingdir

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = asyncio.run(initialize_rag())
    insert_text(rag, unique_contexts)


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()

    with (
        open(output_file, "w", encoding="utf-8") as result_file,
        open(error_file, "w", encoding="utf-8") as err_file,
    ):
        result_file.write("[\n")
        first_entry = True

        for query_text in queries:
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")


def lightrag_query_inference(cfg: DictConfig):
    mode = cfg.project.main_runner.lightrag_inference_mode
    WORKING_DIR = cfg.project.main_runner.light_rag_workingdir
    OUTPUT_PATH = f"{WORKING_DIR}/lightrag_{mode}_result.json"
    ERROR_PATH = f"{WORKING_DIR}/lightrag_{mode}_errors.json"
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
        llm_model_name="granite3.3:8b",  # Your model name
        llm_model_kwargs={"options": {"num_ctx": 8000}},
        # Use Ollama embedding function
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=2000,
            func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text"),
        ),
    )
    query_param = QueryParam(mode=mode, chunk_top_k=5, max_total_tokens=7500)

    queries = extract_queries()
    run_queries_and_save_to_json(queries, rag, query_param, OUTPUT_PATH, ERROR_PATH)
