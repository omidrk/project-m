from mstar.core.graph import EntityGraph, vis_graph
from mstar.schemas.llm_schemas import EntRel, LLMSummary, Relation, Entity
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

# from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.globals import set_llm_cache

set_llm_cache(None)
setup_logging()
logger = logging.getLogger(__name__)


def get_unkonwn_entity_description(
    name: str, desc: str, params: dict, vecstore_objets: OllamaVectorStore
) -> Entity:

    llm_tool = ChatOllama(model=params["model_tools"], cache=False)
    template = load_prompt_template(params["prompt_template_path"])
    prompt = PromptTemplate(
        template=template,
        input_variables=params["prompt_inputs"],
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=Entity)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.debug("Finished preparing get_unkonwn_entity_description LLM configuration.")

    # This is where the LLM will generate the plan and tool calls.
    # The LLM will output a string that contains the tool calls.
    # We need to parse this string and execute the tools.

    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 10:
        encountered_error = False
        retry_counter += 1

        relevant_query = f"name: {name}, description: {desc}"
        # real_data = get_relevant_docs(relevant_query, retriever_summary)
        real_data = vecstore_objets.get_docs_from_store(
            topic="summary", query=relevant_query, merge_content_as_string=True
        )
        try:
            llm_output = llm_chain.invoke(
                {
                    "relation_description": desc,
                    "format_instructions": format_instructions,
                    "real_data": real_data,
                    "name": name,
                }
            )
            res: Entity = fixing_parser.parse(llm_output.content)
            return res
        except Exception as e:
            encountered_error = True
            logger.error(
                f"get_unkonwn_entity_description failed. Entity: {name} Description: {desc}. Error: {e}",
            )


def get_batch_entities(
    entities: list[Entity],
    relations: list[Relation],
    params: dict,
    vectorstore: OllamaVectorStore,
) -> dict:
    ent_dict = {}
    ent_search_list = []

    failed_ents = []
    for ent in entities:
        if ent.name in ent_dict:
            print("Duplicate entity. Pass")
            continue
        ent_dict[ent.name] = ent

    for rel in relations:
        s, r, t, d = (
            rel.source,
            rel.relation,
            rel.target,
            rel.description,
        )
        if not s in ent_dict:
            resolved_entity = get_unkonwn_entity_description(s, d, params, vectorstore)
            if resolved_entity:
                ent_dict[resolved_entity.name] = resolved_entity
            else:
                continue

        if not t in ent_dict:
            resolved_entity = get_unkonwn_entity_description(t, d, params, vectorstore)
            if resolved_entity:
                ent_dict[resolved_entity.name] = resolved_entity
            else:
                continue
    return ent_dict


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def process_ner_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    ner_pipeline_output = load_dict_cache(
        cfg.project.pipelines.NER.cache_ner_dir,
        cfg.project.pipelines.NER.ner_output_name,
    )

    rue_output_name = load_dict_cache(
        cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
        cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
    )
    vectorstore_objects = build_vector_store(ner_pipeline_output, cfg)

    ent_graph_params = {
        "embedder_model": cfg.project.LLM.embedder_name,
        "cache_entity_graph_npz_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_npz_dir,
        "cache_entity_graph_faissindex_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_faissindex_dir,
        "save_attrs": cfg.project.pipelines.entity_graph.ent_graph_cache_attr,
    }
    ent_graph = EntityGraph(dim=cfg.project.LLM.embedder_dim, params=ent_graph_params)
    ent_graph.load_entity_graph()

    get_batch_entities_params = {
        "model_tools": cfg.project.LLM.model_tools,
        "prompt_template_path": cfg.project.pipelines.resolve_unknown_entity.prompt_template_path,
        "prompt_inputs": cfg.project.pipelines.resolve_unknown_entity.prompt_inputs,
    }

    logger.info(f"Loop size is :{len(ner_pipeline_output)}")
    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 10:
        encountered_error = False
        retry_counter += 1
        counter = 0

        for (
            file_name,
            chunk_index,
        ), chunk_dict in ner_pipeline_output.items():

            counter += 1
            if counter > 0 and counter % 10 == 0:
                logger.info(f"{counter} Entity batch processed.")
            if counter > 0 and counter % 30 == 0:
                # save the cache
                save_dict_cache(
                    cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
                    cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
                    rue_output_name,
                )
                ent_graph.save_entity_graph()

            ### for debug
            # if counter > 5:
            #     logger.warning("Exit simulation chunked_summerizer_pipeline...")
            #     save_dict_cache(
            #         cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
            #         cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
            #         rue_output_name,
            #     )
            #     ent_graph.save_entity_graph()
            #     return

            if (file_name, chunk_index) in rue_output_name:
                logger.info(
                    f"Doc Name: {file_name}, Index: {chunk_index} was found in the cache. Skipping."
                )
                continue

            entities: list[Entity] = chunk_dict["entities"]
            relations: list[Relation] = chunk_dict["relations"]
            loop_name_id = {}

            ent_dict_batch = get_batch_entities(
                entities,
                relations,
                vectorstore=vectorstore_objects,
                params=get_batch_entities_params,
            )
            batch_metadata = {"file_name": file_name, "chunk_index": chunk_index}
            for name, ent in ent_dict_batch.items():
                name_id = ent_graph.add_entity(
                    entity=ent, embedding=None, threshold=0.1, metadata=batch_metadata
                )
                loop_name_id[name] = name_id
            for rel in relations:
                s, r, t, d = (
                    rel.source,
                    rel.relation,
                    rel.target,
                    rel.description,
                )
                if s not in loop_name_id or t not in loop_name_id:
                    logger.error(
                        "Source or target entity does not exist in the entity store. Skipping",
                    )
                    continue
                s_id = loop_name_id[s]
                t_id = loop_name_id[t]
                # print('fuckkkk',ent_dict_batch,s_id, r, t_id, d)
                ent_graph.add_relation(s_id, r, t_id, d, batch_metadata)
            rue_output_name[(file_name, chunk_index)] = chunk_dict
    logger.info("Saving final result.")
    save_dict_cache(
        cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
        cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
        rue_output_name,
    )
    ent_graph.save_entity_graph()


def build_vector_store(ner_pipeline_output: dict, cfg: DictConfig) -> OllamaVectorStore:
    ## configuring LLM and Ollama
    vectorstore_objects = OllamaVectorStore(
        embedder_model=cfg.project.LLM.embedder_name
    )
    # building summary docs and content docs for faiss
    summary_doc_list = []
    content_doc_list = []
    for (
        file_name,
        chunk_index,
    ), chunk_dict in ner_pipeline_output.items():

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
    logger.info(f"Doc list size: {len(summary_doc_list)}")
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

    return vectorstore_objects


@hydra.main(config_path="../config/pipelines", config_name="default", version_base=None)
def test_process_ner_pipeline(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded.")

    # Loading caches

    rue_pipeline_output = load_dict_cache(
        cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
        cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
    )
    for k, v in rue_pipeline_output.items():

        print(k, v)
        break

    ent_graph_params = {
        "embedder_model": cfg.project.LLM.embedder_name,
        "cache_entity_graph_npz_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_npz_dir,
        "cache_entity_graph_faissindex_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_faissindex_dir,
        "save_attrs": cfg.project.pipelines.entity_graph.ent_graph_cache_attr,
    }
    ent_graph = EntityGraph(dim=cfg.project.LLM.embedder_dim, params=ent_graph_params)
    ent_graph.load_entity_graph()
    print(
        f"Len Entities: {len(ent_graph.entities)} Len Relations: {len(ent_graph.relations)}"
    )
    ent_pos = ent_graph.pos2id[10]
    sample_ent = ent_graph.entities[ent_pos]
    sample_rel = next(iter(ent_graph.relations.items()))
    print(
        f"sample ent: {sample_ent}, metadata: {ent_graph.id2metadata[ent_pos]} \n\n\n Sample rel: {sample_rel}"
    )
    vis_graph(ent_graph, output_name="BigGraphByType.html", by_type=True)

    vis_graph(ent_graph, output_name="BigGraphByNode.html", by_type=False)
