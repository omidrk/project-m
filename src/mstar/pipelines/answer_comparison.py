import numpy as np
from mstar.core.logger import setup_logging
import logging
from omegaconf import DictConfig
from mstar.core.cache import get_client
import json
import openai

setup_logging()
logger = logging.getLogger(__name__)
client = get_client()


def answer_compare(cfg: DictConfig):

    logger.info("Configuration loaded.")

    openai.api_key = cfg.project.LLM.openai_api_key

    MODE = cfg.project.main_runner.lightrag_inference_mode
    WORKING_DIR = cfg.project.main_runner.light_rag_workingdir
    OTHER_ANSWER_PATH = f"{WORKING_DIR}/lightrag_{MODE}_result.json"
    MSTAR_PATH = f"{cfg.project.pipelines.query_answer.cache_answer_dir}/{cfg.project.pipelines.query_answer.cache_answer_name}.npy"
    FINAL_RESULT_PATH = f"{WORKING_DIR}/eval_{cfg.project.main_runner.answer_1_mode}_{cfg.project.main_runner.answer_2_mode}_result.json"

    with open(OTHER_ANSWER_PATH, "r") as f:
        other_answers = json.load(f)
    mstar_answers = np.load(MSTAR_PATH, allow_pickle=True).item()

    qs = {}
    for k, v in mstar_answers.items():
        qs[k] = {
            "rerank_rev": v["rerank_rev"] if "rerank_rev" in v else "",
            "rag": v["rag"] if "rag" in v else "",
            "rerank_fin": v["rerank_fin"] if "rerank_fin" in v else "",
        }
    for i in other_answers:
        if i["query"] in qs:

            qs[i["query"]][MODE] = i["result"] if "result" in i else ""
    counter = 0
    result = {}

    for q, r in qs.items():

        if q in result:
            continue
        answer_1 = (
            r[MODE]
            if cfg.project.main_runner.answer_1_mode in ["mix", "hybrid"]
            else r["rag"]
        )
        answer_2 = (
            r["rerank_rev"]
            if cfg.project.main_runner.answer_2_mode == "mstar_explanation"
            else r["rerank_fin"]
        )

        sys_prompt = """
                    ---Role---
                    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
                    Each criteria should have a fatctual answer and a data reason. If data is not there consider it wrong. Always weight the data reason. Value short and precise answers.
                    If model is using assumptions without data, or model state there is no data or need access to mode data, reject the answer.
                    """
        prompt = f"""
                    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

                    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
                    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
                    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
                    - **Logicality**: How logically does the answer respond to all parts of the question?
                    - **Relevance**: How relevant is the answer to the question, staying focused and addressing the in tended topic or issue?
                    - **Coherence**: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

                    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

                    Here is the question:
                    {q}

                    Here are the two answers:

                    **Answer 1:**
                    {answer_1}

                    **Answer 2:**
                    {answer_2}

                    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

                    Output your evaluation in the following JSON format:

                    {{
                        "Comprehensiveness": {{
                            "Winner": "[Answer 1 or Answer 2]",
                            "Explanation": "[Provide explanation here]"
                        }},
                        "Diversity": {{
                            "Winner": "[Answer 1 or Answer 2]",
                            "Explanation": "[Provide explanation here]"
                        }},
                        "Empowerment": {{
                            "Winner": "[Answer 1 or Answer 2]",
                            "Explanation": "[Provide explanation here]"
                        }},
                        "Logicality": {{
                            "Winner": "[Answer 1 or Answer 2]",
                            "Explanation": "[Provide explanation here]"
                        }},
                        "Relevance": {{
                            "Winner": "[Answer 1 or Answer 2]",
                            "Explanation": "[Provide explanation here]"
                        }},
                        "Coherence": {{
                            "Winner": "[Answer 1 or Answer 2]",
                            "Explanation": "[Provide explanation here]"
                        }},
                        "Overall Winner": {{
                            "Winner": "[Answer 1 or Answer 2]",
                            "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
                        }}
                    }}
                    """

        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        result_content = response.choices[0].message.content

        # Try parsing output as JSON if the model returns raw JSON string
        try:
            parsed_result = json.loads(result_content)
            # result.append(parsed_result)
            result[q] = parsed_result
            print(f"Counter: {counter}, q: {q}", parsed_result)
            n_counter += 1
        except json.JSONDecodeError as e:
            parsed_result = {"raw_output": result_content}
            print(e, parsed_result)

    with open(FINAL_RESULT_PATH, "w") as f:
        json.dump(result, f)


def process_result(cfg: DictConfig):

    logger.info("Configuration loaded.")

    WORKING_DIR = cfg.project.main_runner.light_rag_workingdir
    FINAL_RESULT_PATH = f"{WORKING_DIR}/eval_{cfg.project.main_runner.answer_1_mode}_{cfg.project.main_runner.answer_2_mode}_result.json"

    with open(FINAL_RESULT_PATH, "r") as f:
        result = json.load(f)

    # Initialize counts per key
    keys = [
        "Comprehensiveness",
        "Diversity",
        "Empowerment",
        "Logicality",
        "Relevance",
        "Coherence",
        "Overall Winner",
    ]
    counts_per_key = {key: {"Answer 1": 0, "Answer 2": 0} for key in keys}

    list_q = [v for k, v in result.items()]

    # Loop through each data object
    for item in list_q:
        for key in keys:
            winner = item[key]["Winner"]
            if winner in counts_per_key[key]:
                counts_per_key[key][winner] += 1
            else:
                print(f"Key not found: {key}")

    # Print results
    print("==== Wins Per Key Across All Data Objects ====")
    for key in keys:
        ps_sum = counts_per_key[key]["Answer 1"] + counts_per_key[key]["Answer 2"]
        ps1 = counts_per_key[key]["Answer 1"] / ps_sum
        ps2 = counts_per_key[key]["Answer 2"] / ps_sum
        print(
            f"{key}: Answer 1 -> {counts_per_key[key]['Answer 1']}, Answer 2 -> {counts_per_key[key]['Answer 2']}"
        )
        print(f"{key}: Answer 1 % -> {ps1 * 100:.2f}, Answer 2 %-> {ps2 * 100:.2f}")
