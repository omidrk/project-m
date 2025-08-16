import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from mstar.core.logger import setup_logging
from mstar.schemas.llm_schemas import Entity, Relation
import faiss
from uuid import uuid4, UUID
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_self_loops
import networkx as nx
from langchain_ollama import OllamaEmbeddings
import torch
from mstar.core.utils import save_dict_to_npz, load_dict_from_npz
from os import path, makedirs

setup_logging()
logger = logging.getLogger(__name__)


class EntityGraph:
    def __init__(self, params: dict, dim: int = 768):
        self.entities: Dict[UUID, Entity] = (
            {}
        )  # Store entities with their embeddings and relations
        self.relations: dict = {}
        self.entity_index = faiss.IndexIDMap2(
            faiss.IndexFlatL2(dim)
        )  # Assuming 768-dimensional embeddings
        self.id2pos: dict = {}
        self.pos2id: dict = {}
        self.next_pos: int = 0
        self.ent_type_label2id: dict = {}
        self.rel_type_label2id: dict = {}
        self.embedder: OllamaEmbeddings = OllamaEmbeddings(
            model=params["embedder_model"]
        )
        self.emb_store: dict = {}

        # To store sum and count
        self.id2Counter: dict = {}
        self.emb_store_sum: dict = {}

        self.params: dict = params
        self.dim: dict = dim
        self.id2metadata: dict[UUID, List[Tuple[str, int]]] = {}

    def save_entity_graph(self):
        data_to_save = {key: getattr(self, key) for key in self.params["save_attrs"]}
        save_dict_to_npz(
            data_to_save,
            path=self.params["cache_entity_graph_npz_dir"],
            filename="entity_graph.npz",
        )
        # Save the faiss index separately
        makedirs(self.params["cache_entity_graph_faissindex_dir"], exist_ok=True)
        if self.entity_index:
            faiss.write_index(
                self.entity_index,
                f'{self.params["cache_entity_graph_faissindex_dir"]}/l2_index.faiss',
            )
            logger.info("Faiss index saved.")

    def load_entity_graph(self):
        dict_value = load_dict_from_npz(
            path=self.params["cache_entity_graph_npz_dir"],
            filename="entity_graph.npz",
        )
        if dict_value:
            # Copy loaded data into the new instance
            for key, value in dict_value.items():
                setattr(self, key, value)
            logger.info("ent_graph npz dict loaded.")
        # Load the faiss index
        if path.exists(
            f'{self.params["cache_entity_graph_faissindex_dir"]}/l2_index.faiss'
        ):
            self.entity_index = faiss.read_index(
                f'{self.params["cache_entity_graph_faissindex_dir"]}/l2_index.faiss'
            )
            logger.info("ent_graph faiss index loaded.")

    def process_ent_label_ids(self):
        if not self.entities:
            return

        type_counter = 0
        for id, ent in self.entities.items():
            if ent.type in self.ent_type_label2id:
                continue
            self.ent_type_label2id[ent.type] = type_counter
            type_counter += 1

    def process_rel_label_ids(self):
        """
        Since relation dict is a list we only consider the first entry.
        TO Do: might need to move to hetero data to include multiple relation between two
        entity.
        """
        if not self.relations:
            return

        type_counter = 0
        for _, rel_list in self.relations.items():
            r = rel_list[0]["relation"]  # consider only first relation.
            if r in self.rel_type_label2id:
                continue
            self.rel_type_label2id[r] = type_counter
            type_counter += 1

    def get_relation_embbeding(
        self,
        source_id: uuid4,
        target_id: uuid4,
        relation: str,
        description: Optional[str],
        return_tensor: bool = False,
    ) -> Union[torch.types.Tensor, list]:
        source_name = self.entities[source_id]
        target_name = self.entities[target_id]
        emb_str = f"{source_name}:{source_id} {relation} {target_name}:{target_id}. Description: {description}"
        emb = self.embedder.embed_documents([emb_str])
        if return_tensor:
            return torch.tensor([emb], dtype=torch.float)
        else:
            return emb

    def get_entity_embbeding(self, ent: Entity, as_np: bool = False) -> Any:
        text = f"name: {ent.name}, description: {ent.description}, type: {ent.type}"
        if as_np:
            return np.array(self.embedder.embed_documents([text]))
        else:
            return self.embedder.embed_documents([text])

    def get_edge_index_n_labels_edgeattr(self):

        self.process_rel_label_ids()
        if not self.relations:
            return
        edge_index = []
        edge_attr = []
        lables = []
        type_counter = 0
        for (s_id, t_id), rel_list in self.relations.items():
            r = rel_list[0]["relation"]  # consider only first relation.
            d = rel_list[0]["description"]
            r_label = self.rel_type_label2id[r]
            s_pos = self.id2pos[s_id]
            t_pos = self.id2pos[t_id]
            lables.append(r_label)
            edge_index.append([s_pos, t_pos])
            edge_attr.append(
                self.get_relation_embbeding(s_id, t_id, r, d, return_tensor=False)
            )

        return (
            torch.tensor(edge_index, dtype=torch.long).T,
            torch.tensor(lables, dtype=torch.long),
            torch.tensor(edge_attr, dtype=torch.float),
        )

    def get_y(self):

        self.process_ent_label_ids()

        if not self.entities:
            return None
        return torch.tensor(
            [self.ent_type_label2id[ent.type] for id, ent in self.entities.items()],
            dtype=torch.long,
        )

    def get_x(self):

        if not self.entities:
            return None
        self.process_ent_label_ids()

        return torch.tensor(
            [self.emb_store[id] for id, _ in self.entities.items()],
            dtype=torch.float,
        )

    def connect_graphs_to_root_index(self, data: Data) -> list[int]:
        G = to_networkx(data, to_undirected=True)
        comp = nx.connected_components(G)
        connect_to_root_indexes = [list(g_set)[0] for g_set in comp]
        return connect_to_root_indexes

    def get_x_edgeindex_edgelabel_y(self):

        x, y = self.get_x(), self.get_y()
        edge_index, labels, attr = self.get_edge_index_n_labels_edgeattr()
        return Data(x=x, edge_index=edge_index, edge_label=labels, y=y, edge_attr=attr)

    def get_rooted_x_edgeindex_edgelabel_y(self):

        x, y = self.get_x(), self.get_y()
        edge_index, labels, edge_attr = self.get_edge_index_n_labels_edgeattr()
        root_index = self.next_pos
        edge_label = max(self.rel_type_label2id.values()) + 1
        connection_indexes = self.connect_graphs_to_root_index(
            Data(x=x, edge_index=edge_index, labels=labels, y=y)
        )
        root_x = torch.zeros_like(x[0]).unsqueeze(dim=0)
        root_x_edge = torch.concatenate([root_x for _ in connection_indexes], dim=0)
        new_indexes = torch.tensor(
            [[root_index, i] for i in connection_indexes], dtype=torch.long
        ).T
        root_edge_labels = torch.tensor(
            [edge_label for i in connection_indexes], dtype=torch.long
        )
        x = torch.concatenate([x, root_x], dim=0)
        y = torch.concatenate([y, torch.tensor([root_index], dtype=torch.long)], dim=0)
        edge_index = torch.concatenate([edge_index, new_indexes], dim=1)
        labels = torch.concatenate([labels, root_edge_labels])
        edge_attrs = torch.concatenate([edge_attr, root_x_edge])
        return Data(
            x=x, edge_index=edge_index, edge_label=labels, y=y, edge_attr=edge_attrs
        )

    def add_entity(
        self,
        embedding: Optional[Union[np.ndarray, None]],
        entity: Entity,
        metadata: Optional[dict],
        threshold: float = 0.1,
    ) -> uuid4:
        """
        Add new entity or merge if similar enough to existing entities
        """
        vec = None
        if not isinstance(embedding, np.ndarray):
            vec = self.get_entity_embbeding(entity, as_np=True)
        # Convert numpy array to C++ structure for Faiss
        vec = vec.reshape(1, -1).astype(np.float32)

        # Search in index
        distances, indices = self.entity_index.search(vec, 5)  # Search top 5 similar

        # Check if any existing entity is close enough
        if len(indices[0]) > 0 and distances[0][0] < threshold:
            # Merge with closest entity
            # print(distances, indices, name, description)
            closest_entity_pos = indices[0][0]
            closest_entity_id = self.pos2id[closest_entity_pos]

            ## average sume for entity embedding.
            old_embbeding = (
                self.emb_store[closest_entity_id].reshape(1, -1).astype(np.float32)
            )
            ## get old embedding emb_sum and counter
            old_embbeding_sum = (
                self.emb_store_sum[closest_entity_id].reshape(1, -1).astype(np.float32)
            )
            old_embbeding_counter = self.id2Counter[closest_entity_id]
            new_embedding_sum = old_embbeding_sum + vec
            new_emb_counter = old_embbeding_counter + 1
            new_embedding = new_embedding_sum / new_emb_counter
            new_embedding = new_embedding.reshape(1, -1).astype(np.float32)
            assert new_embedding.shape == old_embbeding.shape

            self.emb_store[closest_entity_id] = new_embedding
            self.emb_store_sum[closest_entity_id] = new_embedding_sum
            self.id2Counter[closest_entity_id] = new_emb_counter

            self.entity_index.remove_ids(np.array([closest_entity_pos], dtype="int64"))
            self.entity_index.add_with_ids(
                new_embedding, closest_entity_pos
            )  # Add new entity to index

            if metadata:
                if closest_entity_id in self.id2metadata:
                    self.id2metadata[closest_entity_id].append(
                        (metadata["file_name"], metadata["chunk_index"])
                    )
                else:
                    logger.error(
                        "Entity does not have metadata. Check add new entity implementation."
                    )

            logger.info("Entity updated")

            return closest_entity_id

        else:
            # Add as new entity
            new_entity_id = str(uuid4())  # Generate unique ID

            self.entities[new_entity_id] = entity
            self.emb_store[new_entity_id] = vec
            # adding sum and counter
            self.emb_store_sum[new_entity_id] = vec
            self.id2Counter[new_entity_id] = 1

            self.entity_index.add_with_ids(
                vec, self.next_pos
            )  # Add new entity to index
            self.id2pos[new_entity_id] = self.next_pos
            self.pos2id[self.next_pos] = new_entity_id
            self.id2metadata[new_entity_id] = (
                [(metadata["file_name"], metadata["chunk_index"])] if metadata else []
            )

            self.next_pos += 1
            logger.info("New Entity added.")
            return new_entity_id

    def add_relation(
        self,
        source_id: uuid4,
        relation: uuid4,
        target_it: str,
        description: Union[str, None],
        metadata: Optional[dict],
    ):
        if source_id not in self.entities or target_it not in self.entities:
            raise ValueError("Entities must exist before adding relations")
        if (str(source_id), str(target_it)) not in self.relations:
            self.relations[(str(source_id), str(target_it))] = []

        self.relations[(str(source_id), str(target_it))].append(
            {"relation": relation, "description": description, "metadata": metadata}
        )

    def get_similar_entities(self, query_embedding: np.float32, top_k: int = 5):
        """
        Retrieve similar entities based on embedding similarity
        """
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.entity_index.search(vec, top_k)

        similarities = []
        for i in range(len(indices[0])):
            entity_pos = indices[0][i]
            entity_id = self.pos2id[entity_pos]
            similarity_score = 1.0 / (
                distances[0][i] + 1e-8
            )  # Convert distance to similarity
            similarities.append(
                {"entity_id": entity_id, "similarity": similarity_score}
            )

        return sorted(similarities, key=lambda x: -x["similarity"])


def build_nx_graph_by_entity(ent_graph: EntityGraph):
    # Add entities (nodes)
    nx_graph = nx.DiGraph()

    for pos, id in ent_graph.pos2id.items():
        node_ent = ent_graph.entities[id]

        nx_graph.add_node(
            pos, **{"uuid": id, "label": node_ent.name, "node_type": node_ent.type}
        )
    for (s_id, t_id), rels in ent_graph.relations.items():
        s_pos = ent_graph.id2pos[s_id]
        t_pos = ent_graph.id2pos[t_id]
        s_node = ent_graph.entities[s_id]
        t_node = ent_graph.entities[t_id]
        for rel in rels:
            nx_graph.add_edge(
                s_pos,
                t_pos,
                **{
                    "s_id": s_id,
                    "t_id": t_id,
                    "label": rel["relation"],
                    "description": rel["description"],
                },
            )

    return nx_graph


def build_nx_graph_by_type(ent_graph: EntityGraph):

    nx_graph_type = nx.DiGraph()

    for pos, id in ent_graph.pos2id.items():
        node_dict = ent_graph.entities[id]

        if node_dict.type in nx_graph_type:
            continue
        nx_graph_type.add_node(node_dict.type, **{"pos": pos, "label": node_dict.name})

    for (s_id, t_id), rels in ent_graph.relations.items():
        s_pos = ent_graph.id2pos[s_id]
        t_pos = ent_graph.id2pos[t_id]
        s_node = ent_graph.entities[s_id]
        t_node = ent_graph.entities[t_id]
        s_type = s_node.type
        t_type = t_node.type
        # if (s_type,t_type) in self.nx_graph_type:
        #     continue
        for rel in rels:
            nx_graph_type.add_edge(
                s_type,
                t_type,
                **{
                    "s_pos": s_pos,
                    "t_pos": t_pos,
                    "label": rel["relation"],
                    "description": rel["description"],
                },
            )
    return nx_graph_type


def vis_graph(ent_graph: nx.DiGraph, output_name: str, by_type: bool = False):

    import matplotlib.pyplot as plt
    from pyvis.network import Network

    if by_type:
        G = build_nx_graph_by_type(ent_graph)
    else:
        G = build_nx_graph_by_entity(ent_graph)
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
