import json
import time
import pathlib
import networkx as nx
from loguru import logger
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

class FG_Graph_Networkx:
    def __init__(self):
        # 读入merged_dict.json
        with open('data/merged_dict.json', 'r', encoding='utf-8') as f:
            self.merged_dict = json.load(f)
        with open('data/substruct_functional-group.json', 'r', encoding='utf-8') as f:
            self.functional_group_substructure = json.load(f)
        
        def _get_all_functional_groups(merged_dict, functional_groups):
            for k, v in merged_dict.items():
                if isinstance(v, dict):
                    _get_all_functional_groups(v, functional_groups)
                else:
                    functional_groups.append(k)
            return functional_groups        
        self.functional_groups = sorted(list(set(_get_all_functional_groups(self.merged_dict, []))))
        self.G = self._init_graph(self.functional_groups)

    def _init_graph(self, functional_groups):
        G = nx.DiGraph()
        for idx, fg_name in enumerate(functional_groups):
            # G.add_node(fg_name, name=idx)
            G.add_node(fg_name, name=fg_name)
        # 递归地添加边
        def add_edges(merged_dict, G, functional_groups):
            for k, v in merged_dict.items():
                if isinstance(v, dict):
                    for fg_name in v.keys():
                        G.add_edge(k, fg_name)
                    add_edges(v, G, functional_groups)
                else:
                    if len(v) > 0:
                        for fg_name in v:
                            G.add_edge(k, fg_name)
        add_edges(self.merged_dict, G, functional_groups)
        # 检查图是否有环，确保是DAG
        if not nx.is_directed_acyclic_graph(G):
            logger.warning("Graph is not a DAG!")
            # get cycles
            cycles = list(nx.simple_cycles(G))
            logger.debug("Cycles: {}".format(len(cycles)))
            for cycle in cycles:
                logger.debug([functional_groups[i] for i in cycle])
        else:
            logger.debug("Graph is a DAG.")
            topological_order = list(nx.topological_sort(G))
            logger.debug("One possible topological order of the functional groups is:", topological_order)
            G.remove_node("Aromatic")
            G.remove_node("Aliphatic")
            logger.debug("Aromatic and Aliphatic are removed from the graph.")

        # 创建一个副本以避免在迭代时修改图
        G_copy = G.copy()

        # 创建一个边的列表，以便安全地迭代
        edges_to_check = list(G.edges())

        # 检查每一条边
        for u, v in edges_to_check:
            # 移除当前边
            G_copy.remove_edge(u, v)
            # 检查是否还有其他路径从u到v
            if nx.has_path(G_copy, u, v):
                # 如果有其他路径，则原图中这条边是冗余的
                G.remove_edge(u, v)
            # 将边加回到副本中，继续检查下一条边
            G_copy.add_edge(u, v)

        return G

    def get_sub_graph(self, fg_name):
        # 获取所有父节点（祖先）
        ancestors = nx.ancestors(self.G, fg_name)
        # 获取所有子节点（后代）
        descendants = nx.descendants(self.G, fg_name)
        # 合并祖先和后代节点，并包括'carbonyl'本身
        subgraph_nodes = ancestors.union(descendants).union({fg_name})
        # 创建子图
        subgraph = self.G.subgraph(subgraph_nodes)

        plt.clf()
        plt.figure(figsize=(15, 5))
        # 用适合展现DAG层次结构的布局
        pos = graphviz_layout(subgraph, prog='dot')

        # 绘制节点，每个节点上显示节点的name
        nx.draw_networkx_nodes(subgraph, pos, node_size=10, node_color='lightblue')
        # 绘制边，使用半透明的线，较细的线宽
        nx.draw_networkx_edges(subgraph, pos, alpha=0.7, width=0.5, edge_color='grey', style='dashed')
        # 添加标签
        labels = {node: node for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_color='black')

        plt.axis('off')

        pathlib.Path('img').mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        plt.savefig('functional_groups_dag-{}-sub-{}.png'.format(fg_name, now), dpi=600, bbox_inches='tight')

        plt.show()
        # nx.draw(subgraph, with_labels=True)
        # plt.show()

    def visualize_graph(self):
        plt.clf()
        plt.figure(figsize=(125, 25))
        # 用适合展现DAG层次结构的布局
        pos = graphviz_layout(self.G, prog='dot')

        # 绘制节点，每个节点上显示节点的name
        nx.draw_networkx_nodes(self.G, pos, node_size=10, node_color='lightblue')
        # 绘制边，使用半透明的线，较细的线宽
        nx.draw_networkx_edges(self.G, pos, alpha=0.7, width=0.5, edge_color='grey', style='dashed')
        # 添加标签
        labels = {node: self.functional_groups.index(node) for node in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8, font_color='black')

        plt.axis('off')

        pathlib.Path('img').mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        plt.savefig('img/functional_groups_dag-{}.png'.format(now), dpi=300, bbox_inches='tight')

        plt.show()

    def save_to_csv(self):
        with open('functional_groups.csv', 'w') as f:
            f.write('from,to\n')
            for u, v in self.G.edges():
                f.write('"{}","{}"\n'.format(u, v))

# fg_graph = FG_graph()
# fg_graph.visualize_graph()