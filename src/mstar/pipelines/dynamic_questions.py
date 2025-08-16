from mstar.core.utils import load_prompt_template
import numpy as np

from os import path, listdir
from mstar.core.logger import setup_logging
import logging
from omegaconf import DictConfig

from mstar.core.cache import get_client


import numpy as np

from groq import Groq


setup_logging()
logger = logging.getLogger(__name__)
client = get_client()


def question_generation(cfg: DictConfig):

    logger.info("Configuration loaded.")
    gclient = Groq(api_key=cfg.project.LLM.grok_api)

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
    total_description = "\n\n".join(unique_contexts)

    prompt = f"""
    You are given the following dataset description:

    {total_description}

    Your task is to:

    1. **Identify 5 potential user types** who would engage with this dataset.
    2. For each user, list **5 specific tasks** they would perform using this dataset.
    3. For each **(user, task)** combination, generate **5 questions** that:
    - Require a **high-level understanding** of the entire dataset
    - Emphasise **contextual understanding**, multihub reasoning, and semantic relationships
    - Have **short, precise answers**


    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """

    completion = gclient.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a precise and structured assistant.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    WORKING_DIR = cfg.project.main_runner.light_rag_workingdir
    OUTPUT_PATH = f"{WORKING_DIR}/{cfg.project.name}_edited_questions.txt"

    with open(OUTPUT_PATH, "w") as file:
        file.write(completion.choices[0].message.content)

    print(f"questions written to {OUTPUT_PATH}")
