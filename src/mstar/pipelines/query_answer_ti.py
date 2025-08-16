from mstar.core.graph import EntityGraph, vis_graph
from mstar.schemas.llm_schemas import (
    AnswerEvaluator,
    EntRel,
    EntRelb,
    FinalAnswer,
    LLMSummary,
    QueryAnswerLLM,
    QueryReWriter,
    ReRankerScore,
    Relation,
    Entity,
)

from mstar.core.utils import (
    OllamaVectorStore,
    load_dict_cache,
    load_prompt_template,
    save_dict_cache,
    load_eval_qa_csv,
    save_eval_qa_csv,
)
from os import makedirs, path
from mstar.core.logger import setup_logging
import logging
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Tuple, Dict, Union
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import networkx as nx
from functools import reduce
from mstar.core.cache import get_client, redis_cache

# from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.globals import set_llm_cache
import re
import pandas as pd

set_llm_cache(None)

setup_logging()
logger = logging.getLogger(__name__)
logger.info("Lets goooo")
client = get_client()


def get_common_elements(list_of_sets):
    return reduce(lambda x, y: x.intersection(y), list_of_sets)


def get_common_elements_intersection(list_of_sets):
    return reduce(lambda x, y: x.intersection(y), list_of_sets)


def get_common_elements_union(list_of_sets):
    return reduce(lambda x, y: x.union(y), list_of_sets)


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


def return_num_paths(
    final_nodes: set, rel_dict: Dict[int, List[List[Union[int, str]]]]
) -> Tuple[int, int]:
    valid_path = 0
    invalid_paths = 0
    for k, v in rel_dict.items():
        for p_idx, path in enumerate(v):
            if path[-1] in final_nodes:
                valid_path += 1
            else:
                invalid_paths += 1
    print(
        f"Valid paths: {valid_path}, invalid paths: {invalid_paths}, total paths: {valid_path+invalid_paths},num nodes: {len(final_nodes)}"
    )
    return valid_path, invalid_paths


def ner_raw_rag_query(cfg: DictConfig):

    logger.info("Configuration loaded.")
    qs = extract_queries(cfg.project.pipelines.query_answer.ti_question_path)
    logger.info(f"Query loaded. Q total size: {len(qs)}")
    rue_output_name = load_dict_cache(
        cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
        cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
    )

    vectorstore_objects = build_vector_store(rue_output_name, cfg)
    ent_graph_params = {
        "embedder_model": cfg.project.LLM.embedder_name,
        "cache_entity_graph_npz_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_npz_dir,
        "cache_entity_graph_faissindex_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_faissindex_dir,
        "save_attrs": cfg.project.pipelines.entity_graph.ent_graph_cache_attr,
    }
    ent_graph = EntityGraph(dim=cfg.project.LLM.embedder_dim, params=ent_graph_params)
    ent_graph.load_entity_graph()

    G = build_graph_by_type(ent_graph)
    logger.debug(f"Len ent: {len(ent_graph.entities)}, rel: {len(ent_graph.relations)}")

    answer_doc_dict = load_dict_cache(
        cfg.project.pipelines.query_answer.cache_answer_dir,
        cfg.project.pipelines.query_answer.cache_answer_name,
    )
    for idx, q in enumerate(qs):
        if q in answer_doc_dict:
            logger.info("Answer in cache, skip")
            continue
        if idx > 0 and idx % 5 == 0:
            save_dict_cache(
                cfg.project.pipelines.query_answer.cache_answer_dir,
                cfg.project.pipelines.query_answer.cache_answer_name,
                answer_doc_dict,
            )
            logger.info("Saved to storage.")
        logger.info(f"question number: {idx}")
        logger.debug(f"Question is : {q}\n\n")
        answer_doc_dict[q] = {}
        try:
            ent_rel = query_entity_extractor(q, cfg)
            graph_form = ent_rel_to_cnf(ent_rel)
            logger.debug(f"{ent_rel}, {graph_form}")

            ent_dict = {ent.name: ent for ent in ent_rel.entities}

            ## First approach.
            final_nodes, rel_nodes_dict = cnf_similarity_search(
                q,
                ent_rel,
                ent_dict,
                ent_graph,
                G,
                rue_output_name,
                cfg,
                get_similar_entities_num=5,
            )
            logger.debug("Final nodes:\n\n, {final_nodes}, {rel_nodes_dict}")

            rag_answer = naive_rag_query(q, cfg, rue_output_name, 5)

            answer_doc_dict[q]["rag"] = rag_answer.answer
            reached_answer = answer_evaluator(
                q, [rag_answer.answer, rag_answer.think], cfg
            )
            reached_answer = answer_evaluator(
                q, [rag_answer.answer, rag_answer.think], cfg
            )
            if reached_answer:
                logger.info("Reached the final answer by rag:)")
                answer_doc_dict[q]["type_rag"] = 1
                answer_doc_dict[q]["result"] = True

            final_answer, summarylists = reranker_iterative(
                q,
                ent_graph,
                final_nodes,
                cfg,
                rel_nodes_dict,
                rue_output_name,
            )
            revised_ans = reranker_iterative_final_finish(q, final_answer, cfg)
            logger.debug(f"reranker_iterative_final_finish is : {revised_ans}")
            answer_doc_dict[q]["rerank_rev"] = revised_ans
            answer_doc_dict[q]["rerank_fin"] = final_answer

            reached_answer = answer_evaluator(q, summarylists, cfg)
            if reached_answer:
                logger.info("Reached the final answer reranker_iterative:)")
                logger.debug(f"Revised new answer: {revised_ans}")
                answer_doc_dict[q]["type_rerank"] = 1
                answer_doc_dict[q]["result"] = True
                continue
            logger.info("reranker_iterative failed :( trying next method...")
            answer_doc_dict[q]["result"] = False
        except:
            logger.error("Failed q, pass")
            continue
    save_dict_cache(
        cfg.project.pipelines.query_answer.cache_answer_dir,
        cfg.project.pipelines.query_answer.cache_answer_name,
        answer_doc_dict,
    )


def single_ner_raw_rag_query(cfg: DictConfig, q: str = None):

    rue_output_name = load_dict_cache(
        cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
        cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
    )

    ent_graph_params = {
        "embedder_model": cfg.project.LLM.embedder_name,
        "cache_entity_graph_npz_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_npz_dir,
        "cache_entity_graph_faissindex_dir": cfg.project.pipelines.entity_graph.cache_entity_graph_faissindex_dir,
        "save_attrs": cfg.project.pipelines.entity_graph.ent_graph_cache_attr,
    }
    ent_graph = EntityGraph(dim=cfg.project.LLM.embedder_dim, params=ent_graph_params)
    ent_graph.load_entity_graph()

    G = build_graph_by_type(ent_graph)
    logger.info(f"Len ent: {len(ent_graph.entities)}, rel: {len(ent_graph.relations)}")

    if not q:
        q = "What are the figures for [current assets] and [liabilities] of texas instruments[TI] as of [q4] and [q2] and [Q3] of year [2023] and [2024]"
    logger.info(f"Question is : {q}\n\n")

    try:
        ent_rel = query_entity_extractor(q, cfg)
        graph_form = ent_rel_to_cnf(ent_rel)
        logger.debug(f"{ent_rel}, {graph_form}")

        ent_dict = {ent.name: ent for ent in ent_rel.entities}

        ## First approach.
        final_nodes, rel_nodes_dict = cnf_similarity_search(
            q,
            ent_rel,
            ent_dict,
            ent_graph,
            G,
            rue_output_name,
            cfg,
            get_similar_entities_num=2,
        )
        valid_paths, invalid_paths = return_num_paths(final_nodes, rel_nodes_dict)
        logger.debug(f"paths:\n\n {valid_paths}, {invalid_paths}")

        q_graph_path = q[:10].replace(" ", "_")
        H = build_answer_subgraph(ent_graph, rel_nodes_dict, final_nodes, None)
        SEARCHSPACE_PATH = (
            f"{cfg.project.pipelines.query_answer.cache_answer_dir}/{q_graph_path}.html"
        )
        vis_query_graph(H, SEARCHSPACE_PATH)

        rag_answer = naive_rag_query(q, cfg, rue_output_name, 5)
        reached_answer = answer_evaluator(q, [rag_answer.answer, rag_answer.think], cfg)
        reached_answer = answer_evaluator(q, [rag_answer.answer, rag_answer.think], cfg)

        if reached_answer:
            logger.info("Reached the final answer by rag:)")
            logger.info(f"Rag Answer is: {rag_answer}")

        final_answer, summarylists = reranker_iterative(
            q,
            ent_graph,
            final_nodes,
            cfg,
            rel_nodes_dict,
            rue_output_name,
        )
        revised_ans = reranker_iterative_final_finish(q, final_answer, cfg)
        logger.debug(f"reranker_iterative_final_finish is : {revised_ans}")

        reached_answer = answer_evaluator(q, summarylists, cfg)
        if reached_answer:
            logger.info("Reached the final answer reranker_iterative:)")
            logger.debug(f"Revised m* answer:  {revised_ans}")
            logger.debug(f"Original m* answer:  {revised_ans}")

        logger.info("reranker_iterative failed :( trying next method...")
    except:
        logger.error("Failed q, pass")


def ent_rel_to_cnf(ent_rel: EntRel) -> List[List[Tuple[str, str, int]]]:
    ERG = nx.DiGraph()
    ERG.add_nodes_from([ent.name for ent in ent_rel.entities])
    for idx, rel in enumerate(ent_rel.relations):
        ERG.add_edge(rel.source, rel.target, idx=idx)
    # incoming to node can be interpreted as intersection, so the final will be union of intersections.
    unions = []
    for node in ERG.nodes():
        if ERG.in_degree(node) > 0:
            intersections = [
                (s, t, ERG.edges[(s, t)]["idx"]) for s, t in ERG.in_edges(node)
            ]
            unions.append(intersections)
    return unions


def get_cnf_final_nodes(
    unions: List[List[Tuple[str, str, int]]], rel_nodes_dict: dict
) -> set:

    temp_unions = []
    for intersections in unions:
        temp_ansnodes = []
        for s, t, idx in intersections:
            sp = rel_nodes_dict[idx]
            answer_nodes = set([rr[-1] for rr in sp])
            temp_ansnodes.append(answer_nodes)
        intersection_set = get_common_elements_intersection(temp_ansnodes)
        temp_unions.append(intersection_set)
    print(get_common_elements_union(temp_unions))
    return get_common_elements_union(temp_unions)


def cnf_similarity_search(
    query: str,
    ent_rel: EntRel,
    ent_dict: dict,
    ent_graph: EntityGraph,
    G: nx.DiGraph,
    rue_output_name: dict,
    cfg: DictConfig,
    get_similar_entities_num: int = 5,
) -> Tuple[set, dict]:
    rel_nodes_dict = {}
    sim_entity_counter = 0
    found_ans_set = False
    unions = ent_rel_to_cnf(ent_rel)
    while sim_entity_counter < 6 and not found_ans_set:
        get_similar_entities_num += sim_entity_counter * 3
        sim_entity_counter += 1
        logger.info(
            f"Similar item number: {get_similar_entities_num}, counter: {sim_entity_counter}"
        )
        for rel_idx, rel in enumerate(ent_rel.relations):
            if rel.source not in ent_dict or rel.target not in ent_dict:
                logger.warning("Could not find the source or target. Skip rel...")
                continue
            s = ent_dict[rel.source]
            t = ent_dict[rel.target]

            s_emb = ent_graph.get_entity_embbeding(s, as_np=True)
            t_emb = ent_graph.get_entity_embbeding(t, as_np=True)

            sim_ent = ent_graph.get_similar_entities(s_emb, get_similar_entities_num)
            s_nodes = [
                (
                    ent_graph.id2pos[item["entity_id"]],
                    item["similarity"],
                )
                for item in sim_ent
            ]

            sim_ent = ent_graph.get_similar_entities(t_emb, get_similar_entities_num)
            t_nodes = [
                (
                    ent_graph.id2pos[item["entity_id"]],
                    item["similarity"],
                )
                for item in sim_ent
            ]
            sp: List[List[int]] = []
            for node1, _ in s_nodes:
                for node2, _ in t_nodes:
                    try:
                        spp = nx.shortest_path(G, node1, node2)
                        sp.append(spp)

                    except:
                        pass

            rel_nodes_dict[rel_idx] = sp

        final_nodes: set = get_cnf_final_nodes(unions, rel_nodes_dict)
        if final_nodes:
            found_ans_set = True
    if final_nodes:
        logger.info("Get similar entities worked with entities as source.")
        return (final_nodes, rel_nodes_dict)
    else:

        ## use context rag to fetch source entities

        vectorstore_objects = build_vector_store(rue_output_name, cfg)
        rel_nodes_dict = {}
        sim_entity_counter = 0
        get_similar_entities_num = 50
        found_ans_set = False

        for rel_idx, rel in enumerate(ent_rel.relations):
            if rel.source not in ent_dict or rel.target not in ent_dict:
                logger.warning("Could not find the source or target. Skip rel...")
                continue
            s = ent_dict[rel.source]
            t = ent_dict[rel.target]
            r = rel.relation
            d = rel.description

            q_str = f"Entity: name:{s.name},Description: {s.description}. \n Relation:{s.name} {r} {t.name}. Relation description: {d}"
            s_ent_docs = vectorstore_objects.get_docs_from_store(
                "md", q_str, merge_content_as_string=False
            )
            s_nodes = []
            all_s_ents = []
            for doc in s_ent_docs:
                entities = doc.metadata["entities"]
                all_s_ents += entities
            for ent in all_s_ents:
                s_emb = ent_graph.get_entity_embbeding(ent, as_np=True)
                sim_ent = ent_graph.get_similar_entities(s_emb, 1)[0]
                s_nodes.append(
                    (
                        ent_graph.id2pos[sim_ent["entity_id"]],
                        sim_ent["similarity"],
                    )
                )

            t_emb = ent_graph.get_entity_embbeding(t, as_np=True)
            sim_ent = ent_graph.get_similar_entities(t_emb, get_similar_entities_num)
            t_nodes = [
                (
                    ent_graph.id2pos[item["entity_id"]],
                    item["similarity"],
                )
                for item in sim_ent
            ]
            sp: List[List[int]] = []
            for node1, _ in s_nodes:
                for node2, _ in t_nodes:
                    try:
                        spp = nx.shortest_path(G, node1, node2)
                        sp.append(spp)
                    except:
                        pass

            rel_nodes_dict[rel_idx] = sp

        final_nodes: set = get_cnf_final_nodes(unions, rel_nodes_dict)
        logger.info("Get similar entities worked with rag as source.")

        return (final_nodes, rel_nodes_dict)


def get_similarity_set_by_entities(
    query: str,
    ent_rel: EntRel,
    ent_dict: dict,
    ent_graph: EntityGraph,
    G: nx.DiGraph,
    rue_output_name: dict,
    cfg: DictConfig,
) -> Tuple[set, dict]:
    rel_nodes_dict = {}
    answer_nodes_list = []
    sim_entity_counter = 0
    get_similar_entities_num = 15
    found_ans_set = False
    while sim_entity_counter < 6 and not found_ans_set:
        get_similar_entities_num += sim_entity_counter * 3
        sim_entity_counter += 1
        logger.info(
            f"Similar item number: {get_similar_entities_num}, counter: {sim_entity_counter}"
        )
        for rel_idx, rel in enumerate(ent_rel.relations):
            if rel.source not in ent_dict or rel.target not in ent_dict:
                logger.warning("Could not find the source or target. Skip rel...")
                continue
            answer_nodes = set()
            s = ent_dict[rel.source]
            t = ent_dict[rel.target]

            s_emb = ent_graph.get_entity_embbeding(s, as_np=True)
            t_emb = ent_graph.get_entity_embbeding(t, as_np=True)

            sim_ent = ent_graph.get_similar_entities(s_emb, get_similar_entities_num)
            s_nodes = [
                (
                    ent_graph.id2pos[item["entity_id"]],
                    item["similarity"],
                )
                for item in sim_ent
            ]

            sim_ent = ent_graph.get_similar_entities(t_emb, get_similar_entities_num)
            t_nodes = [
                (
                    ent_graph.id2pos[item["entity_id"]],
                    item["similarity"],
                )
                for item in sim_ent
            ]
            sp: List[List[int]] = []
            for node1, _ in s_nodes:
                for node2, _ in t_nodes:
                    try:
                        spp = nx.shortest_path(G, node1, node2)
                        sp.append(spp)

                    except:
                        pass

            rel_nodes_dict[rel_idx] = sp
            answer_nodes = set([rr[-1] for rr in sp])
            answer_nodes_list.append(answer_nodes)
        final_nodes: set = get_common_elements(answer_nodes_list)
        if final_nodes:
            found_ans_set = True
    if final_nodes:
        logger.info("Get similar entities worked with entities as source.")
        return (final_nodes, rel_nodes_dict)
    else:

        ## use context rag to fetch source entities

        vectorstore_objects = build_vector_store(rue_output_name, cfg)
        rel_nodes_dict = {}
        answer_nodes_list = []
        sim_entity_counter = 0
        get_similar_entities_num = 50
        found_ans_set = False

        for rel_idx, rel in enumerate(ent_rel.relations):
            if rel.source not in ent_dict or rel.target not in ent_dict:
                logger.warning("Could not find the source or target. Skip rel...")
                continue
            answer_nodes = set()
            s = ent_dict[rel.source]
            t = ent_dict[rel.target]
            r = rel.relation
            d = rel.description

            q_str = f"Entity: name:{s.name},Description: {s.description}. \n Relation:{s.name} {r} {t.name}. Relation description: {d}"
            s_ent_docs = vectorstore_objects.get_docs_from_store(
                "md", q_str, merge_content_as_string=False
            )
            s_nodes = []
            all_s_ents = []
            for doc in s_ent_docs:
                entities = doc.metadata["entities"]
                all_s_ents += entities
            for ent in all_s_ents:
                s_emb = ent_graph.get_entity_embbeding(ent, as_np=True)
                sim_ent = ent_graph.get_similar_entities(s_emb, 1)[0]
                s_nodes.append(
                    (
                        ent_graph.id2pos[sim_ent["entity_id"]],
                        sim_ent["similarity"],
                    )
                )

            t_emb = ent_graph.get_entity_embbeding(t, as_np=True)
            sim_ent = ent_graph.get_similar_entities(t_emb, get_similar_entities_num)
            t_nodes = [
                (
                    ent_graph.id2pos[item["entity_id"]],
                    item["similarity"],
                )
                for item in sim_ent
            ]
            sp: List[List[int]] = []
            for node1, _ in s_nodes:
                for node2, _ in t_nodes:
                    try:
                        spp = nx.shortest_path(G, node1, node2)
                        sp.append(spp)
                    except:
                        pass
            #     # print(sp)
            #     # return
            rel_nodes_dict[rel_idx] = sp
            answer_nodes = set([rr[-1] for rr in sp])
            answer_nodes_list.append(answer_nodes)
        final_nodes: set = get_common_elements(answer_nodes_list)
        # return
        logger.info("Get similar entities worked with rag as source.")

        return (final_nodes, rel_nodes_dict)


@redis_cache(ttl=30000)
def build_graph_by_type(ent_graph: EntityGraph):

    nx_graph_type = nx.DiGraph()

    for pos, id in ent_graph.pos2id.items():
        ent = ent_graph.entities[id]
        ## adding type and node pos as node and connecting them
        if ent.type not in nx_graph_type.nodes():
            nx_graph_type.add_node(ent.type)
        if pos not in nx_graph_type.nodes():
            nx_graph_type.add_node(
                pos, **{"uuid": ent.uuid, "label": ent.name, "des": ent.description}
            )
        nx_graph_type.add_edge(pos, ent.type)
        nx_graph_type.add_edge(ent.type, pos)
    ### For each real first connect type to type
    ### then conncet node to node directly
    for (s_id, t_id), _ in ent_graph.relations.items():
        s_pos = ent_graph.id2pos[s_id]
        t_pos = ent_graph.id2pos[t_id]
        s_node = ent_graph.entities[s_id]
        t_node = ent_graph.entities[t_id]
        s_type = s_node.type
        t_type = t_node.type

        if s_type not in nx_graph_type.nodes():
            nx_graph_type.add_node(s_type)
        if t_type not in nx_graph_type.nodes():
            nx_graph_type.add_node(t_type)
        if s_pos not in nx_graph_type.nodes():
            nx_graph_type.add_node(s_pos)
        if t_pos not in nx_graph_type.nodes():
            nx_graph_type.add_node(t_pos)
        ## type to type
        if (s_type, t_type) not in nx_graph_type.edges():
            nx_graph_type.add_edge(s_type, t_type)
        ## node to node
        if (s_pos, t_pos) not in nx_graph_type.edges():
            nx_graph_type.add_edge(s_pos, t_pos)

    return nx_graph_type


def build_answer_subgraph(
    ent_graph: EntityGraph,
    rel_nodes_dict: Dict[int, List[list]],
    final_nodes: set,
    top_scores: Optional[List[Tuple[int, str, ReRankerScore]]],
) -> nx.DiGraph:
    H = nx.DiGraph()
    color_map = {}
    colors = [
        "#FF6B6B",  # Coral
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#96CEB4",  # Green
        "#FFEEAD",  # Yellow
        "#D06D8F",  # Purple
        "#8E8C8C",  # Gray
        "#FF9A9E",  # Pink
        "#F3775F",  # Orange
        "#79B4D1",  # Cyan
    ]
    for idx, sp_list in rel_nodes_dict.items():
        for node_chain in sp_list:
            if node_chain[-1] not in final_nodes:
                continue
            pre = None
            for node in node_chain:
                if isinstance(node, str):
                    H.add_node(
                        node,
                        color=colors[idx],
                        **{"label": node},
                    )
                else:
                    ent = ent_graph.entities[ent_graph.pos2id[node]]
                    H.add_node(
                        node,
                        color=colors[idx],
                        **{"label": ent.name},
                        **ent.model_dump(),
                    )
                if pre:
                    H.add_edge(pre, node)
                pre = node
    for node in final_nodes:
        H.nodes[node]["color"] = "#000000"
    if top_scores:
        for node, _, _ in top_scores:
            H.nodes[node]["color"] = "#4CAF50"
    return H


def convert_nx_graph_to_text(G):
    # Get all nodes and their attributes
    nodes = list(G.nodes(data=True))

    # Get all edges (connections)
    edges = list(G.edges())

    # Create node lines with attributes
    node_lines = []
    for node_data in nodes:
        node = node_data[0]
        attrs = node_data[1] if len(node_data) > 1 else {}
        node_lines.append(f"Node: {node}")
        if attrs:

            attrs_str = "[Attributes]:" + ",".join(
                [f"{key}: {value}" for key, value in attrs.items()]
            )
            node_lines.append(attrs_str)
            # for key, value in attrs.items():
            #     node_lines.append(f"{key}: {value}")

    # Create edge lines
    edge_lines = [f"Connection: {src} connects to {tgt}" for src, tgt in edges]

    # Combine all lines
    text_lines = node_lines + [""] + edge_lines

    return "\n".join(text_lines)


def vis_query_graph(G: nx.DiGraph, output_name: str):

    import matplotlib.pyplot as plt
    from pyvis.network import Network

    # Plot with pyvis
    net = Network(
        directed=True,
        select_menu=True,  # Show part 1 in the plot (optional)
        filter_menu=True,  # Show part 2 in the plot (optional)
        neighborhood_highlight=True,
        notebook=True,
    )
    net.show_buttons()  # Show part 3 in the plot (optional)
    net.from_nx(G)  # Create directly from nx graph
    net.show(output_name)


@redis_cache(ttl=900)
def query_entity_extractor(query: str, cfg: DictConfig) -> EntRel:

    domain = cfg.project.domain
    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name, cache=False)
    template = load_prompt_template(
        cfg.project.pipelines.query_answer.prompt_template_path
    )

    prompt_inputs = ["domain", "format_instructions", "query"]
    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=EntRel)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    # print(format_instructions)
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. query_entity_extractor")
    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 4:
        encountered_error = False
        retry_counter += 1
        try:
            llm_output = llm_chain.invoke(
                {
                    "query": query,
                    "format_instructions": format_instructions,
                    "domain": domain,
                }
            )
            res: EntRel = fixing_parser.parse(llm_output.content)

            ent_dict = {ent.name: ent for ent in res.entities}
            for rel_idx, rel in enumerate(res.relations):
                if rel.source not in ent_dict or rel.target not in ent_dict:
                    logger.warning("Could not find the source or target. Retry...")
                    raise

            return res

        except Exception as e:
            encountered_error = True
            logger.error(
                f"NER failed on query: {query}. Error: {e}",
            )


def anwer_node_lastnode_meta(
    answer_set: set, rel_nodes_dict: dict, ent_graph: EntityGraph, rue_output_name: dict
) -> dict:
    """
    Process metadata for nodes in the answer set and generate summary information.

    Args:
        answer_set: Set of answer node identifiers (ints or strings)
        rel_nodes_dict: Dictionary mapping relation nodes to their metadata
        ent_graph: Entity graph containing node metadata
        rue_output_name: Dictionary mapping output names to their metadata

    Returns:
        dict: Dictionary mapping each answer node to its metadata information,
             with the following structure for each node:
                - metadata_set: Set of metadata keys associated with the node
                - meta_summary: String containing summaries of all metadata keys
                - meta_tables: String containing tables from all metadata keys

    Example return structure:
        {
            <answer_node>: {
                "metadata_set": {"key1", "key2"},
                "meta_summary": "Summary for key1\nSummary for key2",
                "meta_tables": "Table1\for key1\nTable1\for key2"
            },
            ...
        }
    """
    query_answer_loop = {}
    for ans_node in answer_set:
        if not isinstance(ans_node, int):
            continue
        node_id = ent_graph.pos2id[ans_node]
        node_meta_list = ent_graph.id2metadata[node_id]
        node_name = ent_graph.entities[node_id].name
        meta_keys = set(node_meta_list)

        meta_summary = "\n".join([rue_output_name[i]["summary"] for i in meta_keys])
        meta_tables = ""
        for i in meta_keys:
            if not rue_output_name[i]["tables"]:
                continue
            meta_tables += "\n".join(
                [v["table"] if v else "" for v in rue_output_name[i]["tables"]]
            )

        query_answer_loop[ans_node] = {
            "metadata_set": meta_keys,
            "meta_summary": meta_summary,
            "meta_tables": meta_tables,
        }
    return query_answer_loop


def lastnode_meta_to_context(path_dict: dict) -> str:
    context = path_dict["meta_summary"] + "\n" + path_dict["meta_tables"]
    return context


def answer_node_path2meta(
    answer_set: set, rel_nodes_dict: dict, ent_graph: EntityGraph, rue_output_name: dict
) -> dict:
    """
    answer:
    {path_idx: {path: [1,2,3], meta_set: {} ,meta_summary: [(),...], meta_table: [], context:'...' }}
    context:
        path: node_a -> node_b -> node_c
        summaries: ...
        tables: ...
        ...
    """
    query_answer_loop = {}
    path_counter = 0
    for _, path_list in rel_nodes_dict.items():
        for path in path_list:
            new_path = []
            meta_to_include: List[Tuple[str, int]] = []
            if path[-1] not in answer_set:
                continue
            for node_pos in path:
                if not isinstance(node_pos, int):
                    new_path.append(node_pos)
                    continue
                node_id = ent_graph.pos2id[node_pos]
                node_meta_list = ent_graph.id2metadata[node_id]
                node_name = ent_graph.entities[node_id].name
                node_des = ent_graph.entities[node_id].description
                new_path.append(node_name)
                meta_to_include += node_meta_list

            meta_keys = set(meta_to_include)
            meta_summary = "\n".join([rue_output_name[i]["summary"] for i in meta_keys])
            meta_tables = ""
            for i in meta_keys:
                if not rue_output_name[i]["tables"]:
                    continue
                meta_tables += "\n".join(
                    [v["table"] if v else "" for v in rue_output_name[i]["tables"]]
                )

            query_answer_loop[path_counter] = {
                "path": new_path,
                "metadata_set": meta_keys,
                "meta_summary": meta_summary,
                "meta_tables": meta_tables,
            }
            path_counter += 1

    return query_answer_loop


def path_dict_to_context(path_dict: dict) -> str:
    path = "->".join(path_dict["path"])
    context = path + "\n" + path_dict["meta_summary"] + "\n" + path_dict["meta_tables"]
    return context


def get_entity_metadata(metastore: OllamaVectorStore, ent: Entity) -> str:
    q = f"name: {ent.name}, type: {ent.type}, description: {ent.description}"
    docs = metastore.get_docs_from_store("summary", q)
    output = ""
    for d in docs:
        output += d.page_content
        output += "\n"
        if "tables" in d.metadata and d.metadata["tables"]:
            tb = "\n".join([val["table"] for val in d.metadata["tables"]])
            output += tb
            output += "\n"
    return output


def get_top_reranker_scores(
    scores_tuple: Dict[str, ReRankerScore], n: int = 1
) -> List[Tuple[str, ReRankerScore]]:
    """Return top N highest ranked scores from the list."""
    sorted_scores = sorted(scores_tuple.items(), key=lambda x: x[1].score, reverse=True)
    return sorted_scores[:n]


# @redis_cache(ttl=30000)
def query_rewriter(query: str, cfg: DictConfig):
    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name, cache=False, seed=42)

    template = """
    Rewrite the query and expand its text to be prepared for the NER extraction.
    clarify dates and abrivations words.
    [FORMAT INSTRUCTIONS]
    {format_instructions}
    [QUERY]
    {query}
    """
    # query:"List all major expense categories reported in Q1 2024 and their amounts."
    # ans: "List all major expense items reported in First Quarter (January - March) 2024 and their total amounts."
    prompt_inputs = ["format_instructions", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )
    pydantic_parser = PydanticOutputParser(pydantic_object=QueryReWriter)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. query_rewriter")

    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 4:
        encountered_error = False
        retry_counter += 1

        try:
            llm_output = llm_chain.invoke(
                {
                    "query": query,
                    "format_instructions": format_instructions,
                }
            )
            res: QueryReWriter = fixing_parser.parse(llm_output.content)
            logger.info("Finished query rewriting.")

            # result.append((context, res))
            # logger.info(f"ent: {ent},Result is: {res}")
            return res.requery

        except Exception as e:
            encountered_error = True
            logger.error(
                f"NER failed on query: {query}. Error: {e}",
            )


def reranker_ansnode_finalnode_only(
    query: str,
    ent_graph: EntityGraph,
    ans_nodes: set,
    cfg: DictConfig,
    vec_store: OllamaVectorStore,
    rel_nodes_dict: dict,
    rue_output_name: dict,
) -> List[Tuple[int, str, ReRankerScore]]:

    # domain = cfg.project.domain
    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name, cache=False, seed=42)
    template = """
        Given the query and the document text below, Rank its relevance on a scale from 0 (no relevance) to 10 (perfect match).
        [QUERY]
        {query} 
        [DOCUMENT]
        {doc}
        [Format Instructions]
        {format_instructions}
        """
    prompt_inputs = ["doc", "format_instructions", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=ReRankerScore)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. reranker_ansnode")
    query_answer_loop = anwer_node_lastnode_meta(
        ans_nodes, rel_nodes_dict, ent_graph, rue_output_name
    )

    cache_context = {}
    for path_id, path_dict in query_answer_loop.items():
        context = lastnode_meta_to_context(path_dict)

        retry_counter = 0
        encountered_error = True
        while encountered_error and retry_counter < 4:
            encountered_error = False
            retry_counter += 1
            if context in cache_context:
                continue
            try:
                llm_output = llm_chain.invoke(
                    {
                        "query": query,
                        "format_instructions": format_instructions,
                        "doc": context,
                    }
                )
                res: ReRankerScore = fixing_parser.parse(llm_output.content)
                cache_context[context] = res

            except Exception as e:
                encountered_error = True
                logger.error(
                    f"NER failed on query: {query}. Error: {e}",
                )
    return cache_context


def answer_evaluator(query: str, summary: List[str], cfg: DictConfig) -> bool:
    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name, cache=False)
    template = """
            Given the query and context below, evaluate whether the context can fully and completly answer the query. 
            If there is no direct answer, try to reason and find answer given tables and context.
            In case of full answer return true else false.
            [CONTEXT]
            {answer}
            [Format Instructions]
            {format_instructions}
            [QUERY]
            {query}
            """
    prompt_inputs = ["answer", "format_instructions", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )
    pydantic_parser = PydanticOutputParser(pydantic_object=AnswerEvaluator)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. answer_evaluator")
    answer = "\n".join(summary) if summary else ""

    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 4:
        encountered_error = False
        retry_counter += 1
        try:
            llm_output = llm_chain.invoke(
                {
                    "query": query,
                    "answer": answer,
                    "format_instructions": format_instructions,
                }
            )
            res: AnswerEvaluator = fixing_parser.parse(llm_output.content)
            logger.info(
                f"Answer evaluator result for {answer} was {res.fully_answered}. +++++++++++Explanation: {res.explanation}"
            )

            return res.fully_answered

        except Exception as e:
            encountered_error = True
            logger.error(
                f"answer_evaluator failed on query: {query}. Error: {e}",
            )


def answer_summarizer(query: str, answer: str, cfg: DictConfig) -> str:

    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name)
    template = """
            Given the query and document text below, provide a concise answer that contains all the relevant text and tables that is a possible answer to part of the query.
            [QUERY]
            {query}
            [CONTEXT]
            {answer}
            """

    prompt_inputs = ["answer", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. answer_summarizer")
    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 4:
        encountered_error = False
        retry_counter += 1
        try:
            llm_output = llm_chain.invoke(
                {
                    "query": query,
                    "answer": answer,
                }
            )
            if llm_output.content:
                return llm_output.content
            else:
                raise

        except Exception as e:
            encountered_error = True
            logger.error(
                f"answer_summarizer failed on query: {query}. Error: {e}",
            )


def get_top_paths_rag(
    q: str, query_answer_loop: dict, cfg: DictConfig, retriever_num: int = 50
) -> List[Document]:
    loop_counter = 0
    vectorstore_objects = OllamaVectorStore(
        embedder_model=cfg.project.LLM.embedder_name
    )
    content_doc_list = []
    for path_id, path_dict in query_answer_loop.items():
        loop_counter += 1
        context = path_dict_to_context(path_dict)

        md_doc = Document(
            page_content=context,
        )
        content_doc_list.append(md_doc)
    logger.info(f"Doc list size: {len(content_doc_list)}")
    vectorstore_objects.gen_vectorstore(
        topic="paths",
        input_docs=content_doc_list,
        retriever_k=retriever_num,
    )
    result = vectorstore_objects.get_docs_from_store("paths", q)
    return result


def reranker_iterative(
    query: str,
    ent_graph: EntityGraph,
    ans_nodes: set,
    cfg: DictConfig,
    rel_nodes_dict: dict,
    rue_output_name: dict,
) -> Tuple[str, List[str]]:
    llm_tool = ChatOllama(model=cfg.project.LLM.model_tools, cache=False)
    template = """
        Given the query and the document text below, Rank its relevance on a scale from 0 (no relevance) to 10 (perfect match).
        [SUMMARY]
        {summary} 
        [DOCUMENT]
        {doc}
        [Format Instructions]
        {format_instructions}
        [QUERY]
        {query}
        """
    prompt_inputs = ["doc", "format_instructions", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=ReRankerScore)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. reranker_iterative")
    query_answer_loop = answer_node_path2meta(
        ans_nodes, rel_nodes_dict, ent_graph, rue_output_name
    )

    cache_context = {}
    answer_summary_cache: List[str] = []
    loop_counter = 0
    last_counter = 0

    for path_id, path_dict in query_answer_loop.items():

        loop_counter += 1
        context = path_dict_to_context(path_dict)
        summary = "\n".join(answer_summary_cache) if answer_summary_cache else ""
        retry_counter = 0
        encountered_error = True
        while encountered_error and retry_counter < 4:
            encountered_error = False
            retry_counter += 1
            if context in cache_context:
                continue
            try:
                llm_output = llm_chain.invoke(
                    {
                        "query": query,
                        "format_instructions": format_instructions,
                        "doc": context,
                        "summary": summary,
                    }
                )
                res: ReRankerScore = fixing_parser.parse(llm_output.content)
                cache_context[context] = res
                if res.score >= 8:
                    print("yeahhhh", f"Loop counter: {loop_counter}")
                    answer_summary_cache.append(res.answer)

                ## check if you reached the answer or not
                print(f"Loop counter: {loop_counter}")
                if loop_counter - last_counter > 5 and len(answer_summary_cache) > 2:
                    print(f"Loop counter: {loop_counter}. Ready for eval result.")
                    last_counter = loop_counter
                    reached_answer = answer_evaluator(query, answer_summary_cache, cfg)
                    reached_answer = answer_evaluator(query, answer_summary_cache, cfg)
                    if reached_answer:
                        logger.info("Reached the final answer :)")
                        return res.answer, answer_summary_cache

            except Exception as e:
                encountered_error = True
                logger.error(
                    f"NER failed on query: {query}. Error: {e}",
                )
    return None, answer_summary_cache


def reranker_iterative_final_finish(
    query: str,
    answer: str,
    cfg: DictConfig,
) -> str:
    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name, cache=False)
    template = """
        Given the query and the document text below, Find the answer of the query only from provided documents. 
        Explain your reasons.
        if you don't know the answer simply respond with, Provided documents can not answer the query.

        [DOCUMENT]
        {doc}
        [QUERY]
        {query}
        """
    prompt_inputs = ["doc", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. reranker_iterative_final_finish")

    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 4:
        encountered_error = False
        retry_counter += 1

        try:
            llm_output = llm_chain.invoke(
                {
                    "query": query,
                    "doc": answer,
                }
            )
            return llm_output.content

        except Exception as e:
            encountered_error = True
            logger.error(
                f"reranker_iterative_final_finish loop failed on query: {query}. Error: {e}",
            )


# @redis_cache(ttl=30000)
def reranker_ansnode(
    query: str,
    ent_graph: EntityGraph,
    ans_nodes: set,
    cfg: DictConfig,
    vec_store: OllamaVectorStore,
    rel_nodes_dict: dict,
    rue_output_name: dict,
) -> List[Tuple[int, str, ReRankerScore]]:

    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name, seed=42)
    template = """
        Given the query and the document text below, Rank its relevance on a scale from 0 (no relevance) to 10 (perfect match).
        [QUERY]
        {query} 
        [DOCUMENT]
        {doc}
        [Format Instructions]
        {format_instructions}
        """
    prompt_inputs = ["doc", "format_instructions", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=ReRankerScore)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration. reranker_ansnode")
    query_answer_loop = answer_node_path2meta(
        ans_nodes, rel_nodes_dict, ent_graph, rue_output_name
    )

    cache_context = {}
    for path_id, path_dict in query_answer_loop.items():
        context = path_dict_to_context(path_dict)
        print("context:::", context)

        retry_counter = 0
        encountered_error = True
        while encountered_error and retry_counter < 4:
            encountered_error = False
            retry_counter += 1
            if context in cache_context:
                continue
            try:
                llm_output = llm_chain.invoke(
                    {
                        "query": query,
                        "format_instructions": format_instructions,
                        "doc": context,
                    }
                )
                res: ReRankerScore = fixing_parser.parse(llm_output.content)
                cache_context[context] = res

            except Exception as e:
                encountered_error = True
                logger.error(
                    f"NER failed on query: {query}. Error: {e}",
                )
    return cache_context


def get_final_answer_from_ranked_nodes(
    query: str,
    scored_nodes: List[Tuple[int, str, ReRankerScore]],
    ent_graph: EntityGraph,
    cfg: DictConfig,
    num_support_doc: int = 3,
) -> List[Tuple[str, str]]:

    llm_tool = OllamaLLM(model=cfg.project.LLM.deep_model_name, seed=42)

    template = """
        Given the query and the document text below, Find the answer of the query only from provided documents.
        if you don't know the answer simply respond with, Provided documents can not answer the query.
        [QUERY]
        {query}
        [DOCUMENTS]
        {doc}
        """
    prompt_inputs = ["doc", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    llm_chain = prompt | llm_tool
    logger.info(
        "Finished preparing LLM configuration. get_final_answer_from_ranked_nodes"
    )
    ans_doc_list = []
    for idx, (node, doc, score_obj) in enumerate(scored_nodes):
        nodes_ids = "Answer ids: "
        ent = ent_graph.entities[ent_graph.pos2id[node]]
        nodes_ids += (
            f"node_id: {ent_graph.pos2id[node]}, relevence_score({score_obj.score})"
        )
        if idx == num_support_doc:
            break
        out = nodes_ids + "\n" + doc

        retry_counter = 0
        encountered_error = True
        while encountered_error and retry_counter < 4:
            encountered_error = False
            retry_counter += 1
            try:
                llm_output = llm_chain.invoke(
                    {
                        "query": query,
                        "doc": out,
                    }
                )
                sp = llm_output.split("</think>")
                think, res = sp[0], sp[1]
                ans_doc_list.append((think, res))
                logger.info(
                    f"fn: get_final_answer_from_ranked_nodes,\n doc:{out} \n think:{think},res:{res}"
                )

            except Exception as e:
                encountered_error = True
                logger.error(
                    f"NER failed on query: {query}. Error: {e}",
                )
    return ans_doc_list


def final_answer_aggregator(
    query: str,
    cfg: DictConfig,
    results: List[Tuple[str, ReRankerScore]],
) -> str:

    llm_tool = OllamaLLM(model=cfg.project.LLM.model_tools, seed=42)

    all_answers = "\n".join(
        [res[1].answer if isinstance(res[1], ReRankerScore) else "" for res in results]
    )
    print("All asnwer is: ")
    template = """
        Given the query and the document text below, Find the answer of the query only from provided documents.
        if you don't know the answer simply respond with, Provided documents can not answer the query.

        [DOCUMENT]
        {doc}
        [QUERY]
        {query}
        """
    prompt_inputs = ["doc", "query"]

    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration . final_answer_aggregator")

    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 4:
        encountered_error = False
        retry_counter += 1
        try:
            llm_output = llm_chain.invoke(
                {
                    "query": query,
                    "doc": all_answers,
                }
            )

            return llm_output

        except Exception as e:
            encountered_error = True
            logger.error(
                f"Final aggregator failed on query: {query}. Error: {e}",
            )


def build_vector_store(rue_output_name: dict, cfg: DictConfig) -> OllamaVectorStore:
    rue_output_name = load_dict_cache(
        cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
        cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
    )
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
    ), chunk_dict in rue_output_name.items():

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


def naive_rag_query(
    query: str, cfg: DictConfig, rue_output_name: dict, num_retrieved: int = 2
) -> QueryAnswerLLM:

    vectorstore_objects = build_vector_store(rue_output_name, cfg)
    support_doc_dict = {}
    answer_doc_list = {}
    logger.info(f"question is: {query}")

    rel_docs = vectorstore_objects.get_docs_from_store(
        "md", query, merge_content_as_string=False
    )
    temp_ans_q = []
    ## include only 2
    for i in range(num_retrieved):
        best_doc = rel_docs[i]
        pg_content = best_doc.page_content
        meta = best_doc.metadata
        if meta["tables"]:
            tables = "\n".join([t["table"] for t in meta["tables"]])
        else:
            tables = ""
        support_str = "\n".join([pg_content, tables])
        temp_ans_q.append(support_str)
    s_doc = "\n".join(temp_ans_q)
    res, llmout = llm_answer_query(query, s_doc, cfg)
    return res


def naive_rag_query_pd(cfg: DictConfig):

    logger.info("Configuration loaded.")

    rue_output_name = load_dict_cache(
        cfg.project.pipelines.resolve_unknown_entity.cache_rue_dir,
        cfg.project.pipelines.resolve_unknown_entity.rue_output_name,
    )

    # del content_doc_list
    vectorstore_objects = build_vector_store(rue_output_name, cfg)
    qa_df = load_eval_qa_csv(
        cfg.project.pipelines.query_answer.eval_file_name,
        cfg.project.pipelines.query_answer.eval_q_path,
    )
    query_dict = qa_df["question"]
    support_doc_dict = {}
    answer_doc_list = {}
    for idx, q in query_dict.items():
        logger.info(f"question number: {idx}")

        rel_docs = vectorstore_objects.get_docs_from_store(
            "md", q, merge_content_as_string=False
        )
        temp_ans_q = []
        ## include only 2
        for i in range(3):
            best_doc = rel_docs[i]
            pg_content = best_doc.page_content
            meta = best_doc.metadata
            if meta["tables"]:
                tables = "\n".join([t["table"] for t in meta["tables"]])
            else:
                tables = ""
            support_str = "\n".join([pg_content, tables])
            temp_ans_q.append(support_str)
        s_doc = "\n".join(temp_ans_q)
        support_doc_dict[idx] = s_doc
        res, llmout = llm_answer_query(q, s_doc, cfg)
        answer_doc_list[idx] = res

    qa_df["support_doc"] = list(support_doc_dict.values())
    qa_df["answers"] = [it.answer for _, it in answer_doc_list.items()]
    qa_df["thinks"] = [it.think for _, it in answer_doc_list.items()]
    logger.info("Finished questions")
    save_eval_qa_csv(
        cfg.project.pipelines.query_answer.eval_result_file_name,
        cfg.project.pipelines.query_answer.eval_q_path,
        qa_df,
    )
    logger.info(
        f"Result saved in csv.{cfg.project.pipelines.query_answer.eval_result_file_name}"
    )


def llm_answer_query(query: str, support_document: str, cfg: DictConfig):
    domain = cfg.project.domain
    llm_tool = ChatOllama(model=cfg.project.LLM.deep_model_name)
    template = """
        Answer the following query only using the support document below.
        ** If answer is not direct, try to reason and find the connections.
        ** If you can not answer, simply answer with No Answer Found. 
        ** Include data and tables supporting the answer.
        [USER QUERY]
        {query}
        [SUPORT DOCUMENT]
        {support_document}
        [FORMAT INSTRUCTION]
        {format_instruction}
        """
    prompt_inputs = ["query", "support_document", "format_instruction"]
    prompt = PromptTemplate(
        template=template,
        input_variables=prompt_inputs,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=QueryAnswerLLM)
    fixing_parser = OutputFixingParser.from_llm(llm=llm_tool, parser=pydantic_parser)
    format_instructions = pydantic_parser.get_format_instructions()
    llm_chain = prompt | llm_tool
    logger.info("Finished preparing LLM configuration.")

    retry_counter = 0
    encountered_error = True
    while encountered_error and retry_counter < 4:
        encountered_error = False
        retry_counter += 1
        try:
            llm_output = llm_chain.invoke(
                {
                    "query": query,
                    "support_document": support_document,
                    "format_instruction": format_instructions,
                }
            )
            res: QueryAnswerLLM = fixing_parser.parse(llm_output.content)

            return res, llm_output

        except Exception as e:
            encountered_error = True
            logger.error(
                f"NER failed on query: {query}. Error: {e}",
            )
