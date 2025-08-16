import logging
from mstar.core.cache import get_client, redis_cache
from mstar.core.logger import setup_logging
from mstar.pipelines.answer_comparison import answer_compare, process_result
from mstar.pipelines.build_graph import (
    process_ner_pipeline,
    test_process_ner_pipeline,
)
from mstar.pipelines.dynamic_questions import question_generation
from mstar.pipelines.lightrag_ti import lightrag_index, lightrag_query_inference
from mstar.pipelines.ner import ner_pipeline, test_ner_pipeline
from mstar.pipelines.process_dirs import (
    process_chunked_tables_pipeline,
    process_pdf_directory_pipeline,
    chunk_pipeline,
    test_process_chunked_tables_pipeline,
)


from mstar.pipelines.query_answer_ti import ner_raw_rag_query, single_ner_raw_rag_query

from mstar.pipelines.summarization import (
    chunked_summerizer_pipeline,
    test_chunked_summerizer_pipeline,
)
import hydra
from omegaconf import DictConfig, OmegaConf

setup_logging()
logger = logging.getLogger(__name__)

client = get_client()


@hydra.main(config_path="config/pipelines", config_name="default", version_base=None)
def main(cfg: DictConfig):

    OmegaConf.resolve(cfg)

    if cfg.project.main_runner.run_main_indexer:
        process_pdf_directory_pipeline(cfg)
        chunk_pipeline(cfg)
        process_chunked_tables_pipeline(cfg)
        test_process_chunked_tables_pipeline(cfg)

        chunked_summerizer_pipeline(cfg)
        test_chunked_summerizer_pipeline(cfg)

        ner_pipeline(cfg)
        test_ner_pipeline(cfg)

        process_ner_pipeline(cfg)
        test_process_ner_pipeline(cfg)

    if cfg.project.main_runner.run_inference:
        logger.info(
            "You are about to run the main inference pipeline. Make sure index pipeline is finished first."
        )

        ner_raw_rag_query(cfg)

    if cfg.project.main_runner.lightrag_index:
        lightrag_index(cfg)

    if cfg.project.main_runner.lightrag_inference:
        lightrag_query_inference(cfg)

    if cfg.project.main_runner.dynamic_question_generation:
        question_generation(cfg)

    if cfg.project.main_runner.answer_comparison:
        answer_compare(cfg)
        process_result(cfg)
    if cfg.project.main_runner.single_query_mstar.enable:

        single_ner_raw_rag_query(cfg, None)


@redis_cache(ttl=300)
def test_logs():
    try:
        logger.debug("Starting application")
        logger.info("Performing some data processing...")

        logger.info("Data processing completed successfully")
        logger.debug("Memory usage: low")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    print("Project is ready")
