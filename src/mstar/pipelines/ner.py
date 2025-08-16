from mstar.schemas.llm_schemas import EntRel, LLMSummary
import numpy as np
from mstar.core.utils import (
    OllamaVectorStore,
    header_spliter,
    load_dict_cache,
    save_dict_cache,
    save_txt,
    load_txt,
    list_files_by_type,
    convert_pdf,
    load_prompt_template,
)
from os import makedirs, path
from mstar.core.logger import setup_logging
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Dict, Union
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_core.globals import set_llm_cache

set_llm_cache(None)

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def ner_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    chunked_summerizer_pipeline_output = load_dict_cache(
        cfg.project.pipelines.chunked_summerizer_pipeline.cache_chunked_summerizer_dir,
        cfg.project.pipelines.chunked_summerizer_pipeline.chunked_summerizer_output_name,
    )

    ner_pipeline_output = load_dict_cache(
        cfg.project.pipelines.NER.cache_ner_dir,
        cfg.project.pipelines.NER.ner_output_name,
    )
    ## configuring LLM and Ollama
    vectorstore_objects = OllamaVectorStore(
        embedder_model=cfg.project.LLM.embedder_name
    )
    ## building summary docs and content docs for faiss
    summary_doc_list = []
    content_doc_list = []

    for (
        file_name,
        chunk_index,
    ), chunk_dict in chunked_summerizer_pipeline_output.items():

        sum_doc = Document(
            page_content=chunk_dict["summary"],
            metadata={"described_md": chunk_dict["described_md"], **chunk_dict},
        )
        md_doc = Document(
            page_content=chunk_dict["described_md"],
            metadata=chunk_dict,
        )
        summary_doc_list.append(sum_doc)
        content_doc_list.append(md_doc)

    if len(summary_doc_list) == 0:
        logger.warning("No document found in the cache. Canceling process...")
        return

    vectorstore_objects.gen_vectorstore(
        topic="summary",
        input_docs=summary_doc_list,
        retriever_k=cfg.project.pipelines.NER.faiss_summary_retriever_k,
    )
    vectorstore_objects.gen_vectorstore(
        topic="md",
        input_docs=content_doc_list,
        retriever_k=cfg.project.pipelines.NER.faiss_md_retriever_k,
    )
    ## free some memory
    del summary_doc_list
    del content_doc_list

    llm_tool = ChatOllama(model=cfg.project.LLM.model_tools, cache=False)
    template = load_prompt_template(cfg.project.pipelines.NER.prompt_template_path)
    prompt = PromptTemplate(
        template=template,
        input_variables=cfg.project.pipelines.NER.prompt_inputs,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=EntRel)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration.")

    domain = cfg.project.domain

    retry_counter = 0
    counter = 0
    encountered_error = True
    logger.info(
        f"Starting Ner extraction. Num Documents: {len(chunked_summerizer_pipeline_output)}"
    )
    while encountered_error and retry_counter < 10:
        encountered_error = False
        retry_counter += 1

        for (
            file_name,
            chunk_index,
        ), chunk_dict in chunked_summerizer_pipeline_output.items():

            counter += 1
            if counter > 0 and counter % 10 == 0:
                logger.info(f"{counter} ner extracted.")
            if counter > 0 and counter % 30 == 0:
                # save the cache
                save_dict_cache(
                    cfg.project.pipelines.NER.cache_ner_dir,
                    cfg.project.pipelines.NER.ner_output_name,
                    ner_pipeline_output,
                )

            ### for debug
            # if counter > 10:
            #     logger.warning("Exit simulation chunked_summerizer_pipeline...")
            #     save_dict_cache(
            #         cfg.project.pipelines.NER.cache_ner_dir,
            #         cfg.project.pipelines.NER.ner_output_name,
            #         ner_pipeline_output,
            #     )
            #     return

            if (file_name, chunk_index) in ner_pipeline_output:
                logger.info(
                    f"Doc Name: {file_name}, Index: {chunk_index} was found in the cache. Skipping."
                )
                continue
            text = chunk_dict["described_md"]  # main text
            relevant_query = ",".join(chunk_dict["topics"])
            support_docs = vectorstore_objects.get_docs_from_store(
                topic="summary", query=relevant_query, merge_content_as_string=True
            )

            try:
                llm_output = llm_chain.invoke(
                    {
                        "support_docs": support_docs,
                        "format_instructions": format_instructions,
                        "real_data": text,
                        "domain": domain,
                    }
                )
                res: EntRel = fixing_parser.parse(llm_output.content)

                chunk_dict["entities"] = res.entities
                chunk_dict["relations"] = res.relations
                ner_pipeline_output[(file_name, chunk_index)] = chunk_dict

            except Exception as e:
                encountered_error = True
                logger.error(
                    f"NER failed on doc: {file_name},index: {chunk_index}. Error: {e}",
                )
    save_dict_cache(
        cfg.project.pipelines.NER.cache_ner_dir,
        cfg.project.pipelines.NER.ner_output_name,
        ner_pipeline_output,
    )


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def test_ner_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    # Loading caches

    ner_pipeline_output = load_dict_cache(
        cfg.project.pipelines.NER.cache_ner_dir,
        cfg.project.pipelines.NER.ner_output_name,
    )
    logger.info(f"Size of the ner object is : {len(ner_pipeline_output)}")
    for k, v in ner_pipeline_output.items():
        if "entities" not in v or "relations" not in v:
            logger.info(f"Found corrupted file: {k}")
        # break

    for k, v in ner_pipeline_output.items():
        print(k, v["entities"], v["relations"])
        break
