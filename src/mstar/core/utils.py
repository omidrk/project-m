from typing import Any, List, Tuple, Union
from os import makedirs, path, listdir
from uuid import uuid4
import httpx

from mstar.core.logger import setup_logging
import logging
from functools import lru_cache
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pandas import DataFrame, read_csv
import hydra
import numpy as np

setup_logging()
logger = logging.getLogger(__name__)


def list_files_by_type(directory: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Lists files in a directory by their file extensions

    Args:
        directory: Path to the directory to scan

    Returns:
        tuple containing three lists:
        - PDF files
        - CSV files
        - All other file types
    Raises:
        FileNotFoundError if directory does not exist
    """
    # Initialize empty lists for each category
    pdf_files: List[str] = []
    csv_files: List[str] = []
    other_files: List[str] = []

    # Check if directory exists
    if not path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Get all entries in the directory
    for entry in listdir(directory):
        full_path = path.join(directory, entry)

        if path.isfile(full_path):  # Ensure it's a file
            extension = path.splitext(entry)[1].lower()

            if extension == ".pdf":
                pdf_files.append(full_path)
            elif extension == ".csv":
                csv_files.append(full_path)
            else:
                other_files.append(full_path)

    return (pdf_files, csv_files, other_files)


async def convert_pdf(filepath: str, params: dict, baseurl: str) -> dict:
    """
    Converts a PDF file to various formats using the docling-serve API.

    Args:
        filepath: The path to the input PDF file.
        params: A dictionary of parameters for the conversion.
        baseurl: The base URL of the docling-serve API.

    Returns:
        A dictionary containing the converted content in different formats.
        The dictionary keys will include 'filename' and keys for the requested
        output formats (e.g., 'md_content', 'json_content').

    Raises:
        httpx.HTTPStatusError: If the API request returns a non-200 status code.
        KeyError: If the API response does not contain the expected 'document' key.
    """
    async_client = httpx.AsyncClient(timeout=60.0)
    url = f"{baseurl}/v1alpha/convert/file"

    files = {
        "files": (path.basename(filepath), open(filepath, "rb"), "application/pdf"),
    }

    try:
        logger.info("Sending post request for PDF process...")
        response = await async_client.post(url, files=files, data=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        if "document" not in data:
            raise KeyError("API response does not contain the 'document' key.")
        logger.info("Request processed.")
        return data

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise
    except KeyError as e:
        logger.error(f"API response parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        await async_client.aclose()


@lru_cache(
    maxsize=None
)  # None means cache size is unbounded, but LRU eviction occurs when memory is needed
def load_prompt_template(template_path: str) -> str:
    """
    Load prompt template from specified path
    Args:
        template_path (str): Path to the prompt file to load
    Returns:
        str: Loaded prompt template
    Raises:
        PromptLoadError: If loading fails
    """
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            loaded_template = f.read()

        logger.info(f"Successfully loaded prompt template from {template_path}")
        return loaded_template

    except FileNotFoundError as e:
        logger.error(f"Prompt template file not found at {template_path}")
        raise Exception("Prompt template file not found")

    except Exception as e:
        logger.error(
            f"Failed to load prompt template at {template_path}. Error: {str(e)}"
        )
        raise Exception(f"Failed to load prompt template: {str(e)}")


def load_txt(file_path):
    """Loads text from a given file path."""
    if path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
        return content
    else:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")


def save_txt(file_path, content):
    """Saves text to a given file path."""
    with open(file_path, "w") as file:
        file.write(content)


def save_dict_cache(stage_dir: str, cache_name: str, file: dict):
    import numpy as np

    try:
        makedirs(stage_dir, exist_ok=True)
        process_directory_cache_save_file = f"{stage_dir}/{cache_name}.npy"
        np.save(process_directory_cache_save_file, file)
        logger.info(f"Cache file - {cache_name} saved.")
    except Exception as e:
        logger.error(f"Failed to save cache file {cache_name}: {str(e)}")
        raise e  # Re-raise the exception after logging


def load_dict_cache(stage_dir: str, cache_name: str) -> dict:
    import numpy as np

    try:
        process_directory_cache_save_file = f"{stage_dir}/{cache_name}.npy"
        if not path.exists(process_directory_cache_save_file):
            logger.info(
                f"Cache file does not exist at {process_directory_cache_save_file}. Skipping {cache_name} cache"
            )
            return {}
        data = np.load(process_directory_cache_save_file, allow_pickle=True)
        logger.info(f"Cache file - {cache_name} loaded.")

        return data.item()
    except Exception as e:
        logger.error(f"Failed to load cache file {cache_name}: {str(e)}")
        raise e


## split based on the headers.
def header_spliter(md_text: str) -> list[str]:
    """
    Chunks a Markdown text into sections based on headers only.

    Args:
    md_text (str): The input markdown text.

    Returns:
    list: A list of strings, each representing a segmented chunk of headers.
    """
    import re

    # Regular expression pattern to match lines starting with '#' followed by optional spaces and text.
    header_pattern = r"^(#+) +(.+)"

    chunks = []
    current_chunk = []
    logger.info("header_spliter started")
    for line in md_text.splitlines():
        match = re.match(header_pattern, line)
        if match:
            level = len(
                match.group(1)
            )  # Header level (e.g., '#' for H1, '##' for H2, etc.)
            current_chunk.append(line)
            if (
                not chunks or chunks[-1][-1] != "\n"
            ):  # Only add a new chunk if it's not the same as the previous one
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
        else:
            current_chunk.append(line)
    logger.info("header_spliter ended")
    return chunks


def save_dict_to_npz(data_dict: Any, filename: str, path: str):
    from numpy import savez

    """
    Saves a Python list to a NumPy .npz file.

    Args:
      data_dict: The list to be saved.
      filename: The name of the .npz file to create.
    """
    try:
        makedirs(path, exist_ok=True)
        save_path = f"{path}/{filename}"
        savez(save_path, data=data_dict)
        logger.info(f"Cache file - {filename} saved.")
    except Exception as e:
        logger.error(f"Failed to save cache file {filename}: {str(e)}")
        raise e  # Re-raise the exception after logging


def load_dict_from_npz(filename: str, path: str):
    from numpy import load
    from os import path as os_path

    """
    Loads a list from a NumPy .npz file.

    Args:
      filename: The name of the .npz file to load from.

    Returns:
      The list that was saved.
    """

    try:
        save_path = f"{path}/{filename}"
        if not os_path.exists(save_path):
            logger.info("No cache found. skipping.")
            return None
        data = load(save_path, allow_pickle=True)
        logger.info(f"Cache file - {filename} saved.")
        return data["data"].item()  # Convert back to a Python list
    except Exception as e:
        logger.error(f"Failed to save cache file {filename}: {str(e)}")
        raise e  # Re-raise the exception after logging


def check_file_exists(filename):
    """
    Checks if a file exists at the given path.

    Args:
      filename: The path to the file to check.

    Returns:
      True if the file exists, False otherwise.
    """
    return path.exists(filename)


def generate_uuid() -> str:
    return str(uuid4())


def normalize_name(name, replacement_dict=None):
    from re import sub

    """
    Normalizes a name by converting to lowercase, removing punctuation,
    and optionally replacing common nicknames with full names.

    Args:
      name: The name to normalize (string).
      replacement_dict: A dictionary where keys are nicknames and values are
                        the full name replacements.  If None, no replacements
                        are made.

    Returns:
      The normalized name (string).
    """
    if isinstance(name, str):  # Handle potential non-string values
        text = name
        # convert to lower case
        text = text.lower()
        text = sub(r"[^a-z]", "", text)
        text = text.strip()

        return text
    else:
        return text  # Return the original value if it's not a string


class OllamaVectorStore:

    def __init__(self, embedder_model: str = "nomic-embed-text"):

        self.embedder = OllamaEmbeddings(model=embedder_model)
        self.faiss_dict: dict = {}

    def gen_vectorstore(self, topic: str, input_docs: List[Document], retriever_k: int):
        v_store = FAISS.from_documents(input_docs, self.embedder)
        self.faiss_dict[topic] = {
            "vs": v_store,
            "rt": v_store.as_retriever(search_kwargs={"k": retriever_k}),
        }
        logger.info(f"Topic: {topic} added to vector store.")

    def get_docs_from_store(
        self, topic: str, query: str, merge_content_as_string: bool = False
    ) -> Union[List[Document], str]:
        if topic not in self.faiss_dict:
            logger.error(f"No vectorstore found for the topic: {topic}")
            return []
        # v_store = self.faiss_dict[topic]["vs"]
        retriever = self.faiss_dict[topic]["rt"]
        res = retriever.invoke(query)
        if merge_content_as_string:
            return "\n".join([doc.page_content for doc in res])
        return res


def load_eval_qa_csv(name: str, base_path: str) -> DataFrame:
    full_path = path.join(base_path, name)
    return read_csv(full_path)  # Load CSV file using pandas read_csv


def save_eval_qa_csv(name: str, base_path: str, df: DataFrame) -> DataFrame:
    full_path = path.join(base_path, name)
    return df.to_csv(full_path)  # Load CSV file using pandas read_csv
