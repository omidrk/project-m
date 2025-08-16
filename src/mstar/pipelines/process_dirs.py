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
from typing import List, Tuple, Dict, Union
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from langchain_core.globals import set_llm_cache

set_llm_cache(None)

setup_logging()
logger = logging.getLogger(__name__)


def process_directory(
    pdf_dir: str,
    baseurl: str,
    params: dict,
    stage_dir: str,
    cache: dict,  # key:str = file_name, value:tuple = (processed_path,status: ['success','fail'])
    stageing: bool = True,
) -> tuple[dict, int]:
    """
    Process PDF files from input directory with optional staging.

    Args:
        pdf_dir: Input directory containing PDF files
        baseurl: Base URL for processing
        params: Additional parameters for processing
        stageing: Whether to use staging process
        stage_dir: Directory for staged files

    Returns:
        Tuple containing lists of successful and failed processing file names
    """

    ## To do: add caching

    # Get list of PDF files
    pdf_names, _, _ = list_files_by_type(pdf_dir)
    logger.info("Started processing directory.")
    successful = []
    failed = []
    failed_counter = 0
    # Ensure the output directory exists
    makedirs(stage_dir, exist_ok=True)

    for file_path in pdf_names:
        # Check existing processed file in staging directory
        ## get the file name

        file_name_ext = file_path.split("/")[-1]
        file_name = path.splitext(file_name_ext)[0]
        processed_path = path.join(stage_dir, f"stage_1_{file_name}.npy")
        if (
            file_name_ext in cache
            and cache[file_name_ext][1] == "success"
            and stageing
            and path.exists(processed_path)
        ):
            continue
        try:
            cache[file_name_ext] = (processed_path, "success")  ## just for test
            logger.info(f"{file_name_ext} processed.")

        except Exception as e:
            failed.append(processed_path)
            cache[file_name_ext] = (processed_path, "fail")
            logger.error(f"Error processing {processed_path}: {str(e)}")
            failed_counter += 1

    return cache, failed_counter


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def process_pdf_directory_pipeline(cfg: DictConfig, use_cache: bool = True):
    process_directory_cache = {}

    if use_cache:
        process_directory_cache = load_dict_cache(
            cfg.project.pipelines.extract_pdf.stage_dir,
            cfg.project.pipelines.extract_pdf.cache_file_name,
        )
    logger.info("Starting process_pdf_directory.")
    stage1_processed, failed_number = process_directory(
        pdf_dir=cfg.project.pipelines.extract_pdf.pdf_dir,
        stage_dir=cfg.project.pipelines.extract_pdf.stage_dir,
        baseurl=None,
        cache=process_directory_cache,
        # lineage_graph=data_linage,
        params={},
    )
    logger.info("Finished process_pdf_directory.")
    save_dict_cache(
        cfg.project.pipelines.extract_pdf.stage_dir,
        cfg.project.pipelines.extract_pdf.cache_file_name,
        process_directory_cache,
    )


def cache_chunks(
    file_name: str,
    chunk_list: list[str],
    # data_lineage: DataLineageGraph,
    cache_path: str,
) -> tuple[list[str], list[str]]:
    if not path.exists(cache_path):
        try:
            makedirs(cache_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create cache directory {cache_path}: {str(e)}")
            raise

    chunked_files_path = []
    chunked_node_names = []
    logger.info("cache_chunks Started.")

    try:
        for idx, chunk in enumerate(chunk_list):
            chunk_name = f"{file_name}.chunk_{idx}"
            store_path = f"{cache_path}/{chunk_name}"

            try:
                save_txt(store_path, chunk)
                chunked_files_path.append(store_path)
                chunked_node_names.append(chunk_name)
                logger.info(f"Successfully saved chunk {chunk_name}")

            except Exception as e:
                logger.error(f"Failed to save chunk {chunk_name}: {str(e)}")
                raise  # Re-raise the exception after logging

    except Exception as e:
        logger.error(f"Error processing chunks: {str(e)}")
        raise

    return chunked_files_path, chunked_node_names


def md_header_split(
    md_doc: str, as_text_list: bool = True
) -> Union[List[str], List[Document]]:
    """Split markdown document into sections based on headers.

    Args:
        md_doc (str): Markdown content to be split
        as_text_list (bool, optional): Whether to return plain text list or
            Document objects. Defaults to True.

    Returns:
        Union[List[str], List[Document]]: Content sections, either as strings
            or as Document if as_text_list is False.
    """
    ## prepare header spliter
    headers_to_split = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdownspliter = MarkdownHeaderTextSplitter(
        headers_to_split, return_each_line=False, strip_headers=False
    )
    splits = markdownspliter.split_text(md_doc)
    ## to return as list
    if as_text_list:
        return [i.page_content for i in splits]
    else:
        splits


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def chunk_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    process_directory_full_path = f"{cfg.project.pipelines.extract_pdf.stage_dir}/{cfg.project.pipelines.extract_pdf.cache_file_name}.npy"
    if not path.exists(process_directory_full_path):
        logger.warning(
            "There is no process_directory_cache. Run the process_pdf_directory pipeline first."
        )
        return
    process_directory_cache = load_dict_cache(
        cfg.project.pipelines.extract_pdf.stage_dir,
        cfg.project.pipelines.extract_pdf.cache_file_name,
    )
    makedirs(cfg.project.pipelines.split_chunk_files.cache_chunks_dir, exist_ok=True)

    logger.info("starting chunk_cache_pipeline loop.")
    encountered_error = True
    error_count = 0
    chunk_pipeline_output = {}
    ## load cache if exist
    chunk_pipeline_output = load_dict_cache(
        cfg.project.pipelines.split_chunk_files.cache_chunks_dir,
        cfg.project.pipelines.split_chunk_files.chunk_pipeline_output_name,
    )
    while encountered_error and error_count < 3:
        encountered_error = False
        for k, (prc_path, status) in process_directory_cache.items():
            if status == "fail":
                continue
            if k in chunk_pipeline_output:
                continue

            try:
                doc_dict: dict = np.load(prc_path, allow_pickle=True).item()
                doc_md = doc_dict["document"]["md_content"]
                # chunks = header_spliter(doc_md)
                chunks = md_header_split(doc_md)
                # print(chunks)
                # return
                cache_chunks_dir = (
                    cfg.project.pipelines.split_chunk_files.cache_chunks_dir
                )
                chunked_files_path, chunked_names = cache_chunks(
                    k, chunks, cache_chunks_dir
                )
                chunk_pipeline_output[k] = chunked_files_path

            except (IOError, ValueError) as e:
                logger.error(f"Error loading document {k}: {e}")
                encountered_error = True
                error_count += 1
                continue
    logger.info("finished chunk_cache_pipeline loop.")
    save_dict_cache(
        cfg.project.pipelines.split_chunk_files.cache_chunks_dir,
        cfg.project.pipelines.split_chunk_files.chunk_pipeline_output_name,
        chunk_pipeline_output,
    )


def find_tables(text: str) -> list[dict]:
    tables = []
    lines = text.split("\n")

    current_table = []
    in_table = False
    start_line = 0

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Check if this could be the start of a table (header row)
        if "|" in stripped_line and (
            "---" in stripped_line or any(cell in stripped_line for cell in ["|", "||"])
        ):
            if not in_table:
                start_line = i
                in_table = True

        # If we're inside a table, add lines to current_table until the end of the table
        if in_table:
            current_table.append(line)

            # Check if this line indicates the end of the table (no more pipes or empty line)
            if "|" not in stripped_line and len(stripped_line) > 0:
                end_line = i
                tables.append(
                    {
                        "start_line": start_line + 1,  # Converting to 1-based index
                        "end_line": end_line,
                        "table": "\n".join(current_table),
                    }
                )
                current_table = []
                in_table = False

        # If we're not in a table and encounter a line with pipes, check if it's the start of a new table
        else:
            if "|" in stripped_line:
                # This could potentially be a table header - reevaluate
                pass  # Note: This might require more sophisticated logic for perfect detection

    return tables


def describe_table_llm(table: str, config: dict) -> str:
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate

    # print(config.resolve_parse_tree())

    try:
        # Initialize LLM and prompt chain
        template = load_prompt_template(config["prompt_template_path"])
        llm = OllamaLLM(model=config["model_name"], cache=False)
        prompt = PromptTemplate(
            template=template,
            input_variables=[config["template_input_name"]],
        )
        chain = prompt | llm

        # Log invocation
        logger.debug(f"Invoking LLM with table: {table}")
        llm_output = chain.invoke({config["template_input_name"]: table})

        # Process output based on config
        if config["think"]:
            logger.debug("Processing LLM output")
            result = llm_output.split("think")[-1]

        return result

    except Exception as e:
        logger.error(f"Error in describe_table_llm: {str(e)}", exc_info=True)
        return None


def remove_lines_and_replace(
    md_table: List[str],
    start_line: int,
    end_line: int,
    new_content: str,
    offset: int = 0,
    is_file: bool = False,
) -> Tuple[List[str], int]:
    try:
        logging.info(f"Starting to replace lines from {start_line} to {end_line}")

        if is_file:
            raise NotImplementedError

        new_lines = new_content.split("\n")
        modified_table = (
            md_table[: start_line + offset - 1]
            + new_lines
            + md_table[end_line + offset :]
        )
        logging.info("Successfully replaced lines in table")

        offset = len(new_lines) - (end_line - start_line) + offset - 1
        return modified_table, offset

    except ValueError as e:
        logging.error(f"Error modifying table: {e}")
        raise
    except Exception as e:  # Catch other unexpected exceptions
        logging.error(f"Unexpected error during line replacement: {e}", exc_info=True)
        raise


def process_table_and_describe(
    md_text: str, table_list: List[dict], cache: dict, cfg: DictConfig
) -> str:
    try:
        md_lines = md_text.split("\n")
        global_offset = 0
        counter = 0
        # Convert to a standard dictionary
        # describe_table_config = OmegaConf.to_dict(resolved_config)
        # print("asdjASjd ", resolved_config)
        describe_table_config = (
            cfg.project.pipelines.process_chunked_tables_pipeline.describe_table_llm
        )
        logger.info(f"Processing table...")

        for i, table in enumerate(table_list):
            try:
                if (
                    "start_line" not in table
                    or "end_line" not in table
                    or "table" not in table
                ):
                    logger.warning("Malformed table. Skip processing table.")
                    continue

                if (table["start_line"], table["end_line"]) in cache:
                    llm_out = cache[(table["start_line"], table["end_line"])][
                        "response"
                    ]

                else:
                    llm_out = describe_table_llm(table["table"], describe_table_config)

                md_lines, global_offset = remove_lines_and_replace(
                    md_lines,
                    table["start_line"],
                    table["end_line"],
                    llm_out,
                    global_offset,
                )

                cache[(table["start_line"], table["end_line"])] = {
                    "table": table["table"],
                    "response": llm_out,
                }

                counter += 1
                if counter % 3 == 0:
                    logger.info(f"table progress: {counter} table processed.")

            except Exception as e:
                logger.error(
                    f"Error processing table at lines {table.get('start_line')} - {table.get('end_line')}: {str(e)}"
                )

        return "\n".join(md_lines)

    except Exception as e:
        logger.error("Main processing error:", exc_info=True)
        raise


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def process_chunked_tables_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    # Loading caches
    chunk_pipeline_output = load_dict_cache(
        cfg.project.pipelines.split_chunk_files.cache_chunks_dir,
        cfg.project.pipelines.split_chunk_files.chunk_pipeline_output_name,
    )

    ## prepare output:

    chunked_tables_pipeline_output = load_dict_cache(
        cfg.project.pipelines.process_chunked_tables_pipeline.cache_process_chunked_tables_dir,
        cfg.project.pipelines.process_chunked_tables_pipeline.process_chunked_tables_output_name,
    )

    for_loop_log_str = "File: {}, chunk_index: {}, Process: {}, Status: {}"
    failed_number = 0
    counter = 0
    logger.info(f"Starting loop over files. Loop size: {len(chunk_pipeline_output)}")
    for name, chunk_path_list in chunk_pipeline_output.items():
        for ch_idx, ch in enumerate(chunk_path_list):
            counter += 1
            if counter > 0 and counter % 10 == 0:
                logger.info(f"{counter} chunk processed.")
            if counter > 0 and counter % 30 == 0:
                # save the cache
                save_dict_cache(
                    cfg.project.pipelines.process_chunked_tables_pipeline.cache_process_chunked_tables_dir,
                    cfg.project.pipelines.process_chunked_tables_pipeline.process_chunked_tables_output_name,
                    chunked_tables_pipeline_output,
                )

            ### for debug
            # if counter > 500:
            #     logger.warning("Exit simulation...")
            #     save_dict_cache(
            #         cfg.project.pipelines.process_chunked_tables_pipeline.cache_process_chunked_tables_dir,
            #         cfg.project.pipelines.process_chunked_tables_pipeline.process_chunked_tables_output_name,
            #         chunked_tables_pipeline_output,
            #     )
            #     return

            chunk = load_txt(ch)
            if (name, ch_idx) in chunked_tables_pipeline_output:
                logger.info(f"file {name}, index {ch_idx} found in cache. Skipping...")
                continue
            try:
                table_line_dict = find_tables(chunk)
                # print("\nnnnnn ", table_line_dict)
                if not table_line_dict:
                    logger.info(
                        for_loop_log_str.format(
                            name, ch_idx, "find_tables", "Found no table."
                        )
                    )
                    chunked_tables_pipeline_output[(name, ch_idx)] = {
                        "file_name": name,
                        "chunk_index": ch_idx,
                        "chunk_path": ch,
                        "described_md": chunk,
                        "tables": None,
                    }
                    continue
                logger.info(
                    for_loop_log_str.format(
                        name, ch_idx, "find_tables", "Found tables."
                    )
                )

                logger.info("Starting process_table_and_describe")
                described_md = process_table_and_describe(
                    md_text=chunk,
                    table_list=table_line_dict,
                    cache={},
                    cfg=cfg,
                )
                chunked_tables_pipeline_output[(name, ch_idx)] = {
                    "file_name": name,
                    "chunk_index": ch_idx,
                    "chunk_path": ch,
                    "described_md": described_md,
                    "tables": table_line_dict,
                }

            except Exception as e:
                logger.error(
                    f"Failed to process name: {name}, index:{ch} :=> error: {e}"
                )
                failed_number += 1
                continue
        logger.info(f"File {name} chunks finished.")
    logger.info(
        f"Loop over files finished. Counter is at: {counter}, Failed number: {failed_number},"
    )
    logger.info("Saving output to cache...")
    save_dict_cache(
        cfg.project.pipelines.process_chunked_tables_pipeline.cache_process_chunked_tables_dir,
        cfg.project.pipelines.process_chunked_tables_pipeline.process_chunked_tables_output_name,
        chunked_tables_pipeline_output,
    )


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def test_process_chunked_tables_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    # Loading caches
    chunk_pipeline_output = load_dict_cache(
        cfg.project.pipelines.split_chunk_files.cache_chunks_dir,
        cfg.project.pipelines.split_chunk_files.chunk_pipeline_output_name,
    )

    ## prepare output:

    chunked_tables_pipeline_output = load_dict_cache(
        cfg.project.pipelines.process_chunked_tables_pipeline.cache_process_chunked_tables_dir,
        cfg.project.pipelines.process_chunked_tables_pipeline.process_chunked_tables_output_name,
    )
    logger.info(
        f"chunked_tables_pipeline_output size is : {len(chunked_tables_pipeline_output)}"
    )
    for k, v in chunked_tables_pipeline_output.items():

        print(v)
        break
