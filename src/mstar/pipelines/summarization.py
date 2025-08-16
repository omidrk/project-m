from mstar.schemas.llm_schemas import LLMSummary
import numpy as np
from mstar.core.utils import (
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
from typing import List, Tuple, Dict
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.docstore.document import Document
from langchain_core.globals import set_llm_cache

set_llm_cache(None)

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def chunked_summerizer_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    chunked_tables_pipeline_output = load_dict_cache(
        cfg.project.pipelines.process_chunked_tables_pipeline.cache_process_chunked_tables_dir,
        cfg.project.pipelines.process_chunked_tables_pipeline.process_chunked_tables_output_name,
    )
    chunked_summerizer_pipeline_output = load_dict_cache(
        cfg.project.pipelines.chunked_summerizer_pipeline.cache_chunked_summerizer_dir,
        cfg.project.pipelines.chunked_summerizer_pipeline.chunked_summerizer_output_name,
    )
    ## configuring LLM and Ollama

    llm_tool = ChatOllama(model=cfg.project.LLM.model_tools, cache=False)
    template = load_prompt_template(
        cfg.project.pipelines.chunked_summerizer_pipeline.prompt_template_path
    )
    prompt = PromptTemplate(
        template=template,
        input_variables=cfg.project.pipelines.chunked_summerizer_pipeline.prompt_inputs,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=LLMSummary)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration.")
    retry_counter = 0
    counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 10:
        encountered_error = False
        retry_counter += 1
        for (
            file_name,
            chunk_index,
        ), chunk_dict in chunked_tables_pipeline_output.items():

            counter += 1
            if counter > 0 and counter % 10 == 0:
                logger.info(f"{counter} summary processed.")
            if counter > 0 and counter % 30 == 0:
                # save the cache
                save_dict_cache(
                    cfg.project.pipelines.chunked_summerizer_pipeline.cache_chunked_summerizer_dir,
                    cfg.project.pipelines.chunked_summerizer_pipeline.chunked_summerizer_output_name,
                    chunked_summerizer_pipeline_output,
                )

            ### for debug
            # if counter > 10:
            #     logger.warning("Exit simulation chunked_summerizer_pipeline...")
            #     save_dict_cache(
            #         cfg.project.pipelines.chunked_summerizer_pipeline.cache_chunked_summerizer_dir,
            #         cfg.project.pipelines.chunked_summerizer_pipeline.chunked_summerizer_output_name,
            #         chunked_summerizer_pipeline_output,
            #     )
            #     return

            if (file_name, chunk_index) in chunked_summerizer_pipeline_output:
                logger.info(
                    f"Doc Name: {file_name}, Index: {chunk_index} was found in the cache. Skipping."
                )
                continue

            metadata = {
                "file_name": file_name,
                "chunk_index": chunk_index,
                "chunk_path": chunk_dict["chunk_path"],
                "tables": chunk_dict["tables"],
                "described_md": chunk_dict["described_md"],
            }

            try:
                llm_output = llm_chain.invoke(
                    {
                        "content": chunk_dict["described_md"],
                        "format_instructions": format_instructions,
                    }
                )
                res: LLMSummary = fixing_parser.parse(llm_output.content)
                # summary = LLMSummary(*res)
                summary_content = f"{res.summary}\n{','.join(res.topics)}"
                metadata["topics"] = res.topics
                metadata["summary"] = summary_content
                chunked_summerizer_pipeline_output[(file_name, chunk_index)] = metadata

            except Exception as e:
                encountered_error = True
                logger.error(
                    f"Summarization failed on doc: {file_name},index: {chunk_index}. Error: {e}",
                )
    save_dict_cache(
        cfg.project.pipelines.chunked_summerizer_pipeline.cache_chunked_summerizer_dir,
        cfg.project.pipelines.chunked_summerizer_pipeline.chunked_summerizer_output_name,
        chunked_summerizer_pipeline_output,
    )


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def test_chunked_summerizer_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    # Loading caches
    chunked_summerizer_pipeline_output = load_dict_cache(
        cfg.project.pipelines.chunked_summerizer_pipeline.cache_chunked_summerizer_dir,
        cfg.project.pipelines.chunked_summerizer_pipeline.chunked_summerizer_output_name,
    )
    for k, v in chunked_summerizer_pipeline_output.items():

        print(k, v["topics"], v["summary"])
        break
