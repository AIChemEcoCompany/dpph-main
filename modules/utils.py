from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

def draw_chemical_reaction(smiles, highlightByReactant=False, font_scale=1.5):
    rxn = rdChemReactions.ReactionFromSmarts(smiles,useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)
    # move atom maps to be annotations:
    for m in trxn.GetReactants():
        moveAtomMapsToNotes(m)
    for m in trxn.GetProducts():
        moveAtomMapsToNotes(m)
    d2d = rdMolDraw2D.MolDraw2DSVG(800,300)
    d2d.drawOptions().annotationFontScale=font_scale
    d2d.DrawReaction(trxn,highlightByReactant=highlightByReactant)

    d2d.FinishDrawing()

    return d2d.GetDrawingText()

def moveAtomMapsToNotes(m):
    for at in m.GetAtoms():
        if at.GetAtomMapNum():
            at.SetProp("atomNote",str(at.GetAtomMapNum()))


from rdkit import Chem
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.algorithms import isomorphism


def smarts_to_graph(smarts: str):
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return None
    G = nx.Graph()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        G.add_node(idx, smarts=atom.GetSmarts())
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(a1, a2, smarts=bond.GetSmarts())
    return G


def canonical_smarts_hash(smarts: str):
    G = smarts_to_graph(smarts)
    if G is None:
        return None
    return weisfeiler_lehman_graph_hash(G, edge_attr="smarts", node_attr="smarts")


def are_graphs_isomorphic(G1, G2):
    nm = isomorphism.categorical_node_match("smarts", None)
    em = isomorphism.categorical_edge_match("smarts", None)
    GM = isomorphism.GraphMatcher(G1, G2, node_match=nm, edge_match=em)
    return GM.is_isomorphic()


class SmartsSet:
    def __init__(self, verbose=False):
        self._data = {}   # 哈希桶：hash -> list of (graph, smarts)
        self._order = []  # 保持插入顺序
        self._logs = []   # 日志记录
        self.verbose = verbose

    def _log(self, message: str):
        self._logs.append(message)
        if self.verbose:
            print(message)

    def add(self, smarts: str) -> bool:
        """尝试加入一个 SMARTS，返回是否成功（去重后只存一次）"""
        G = smarts_to_graph(smarts)
        if G is None:
            self._log(f"[FAIL] 无法解析 SMARTS: {smarts}")
            return False

        h = canonical_smarts_hash(smarts)
        if h not in self._data:
            self._data[h] = [(G, smarts)]
            self._order.append(smarts)
            self._log(f"[ADD] 新增 SMARTS: {smarts}")
            return True
        else:
            for G_old, sma_old in self._data[h]:
                if are_graphs_isomorphic(G, G_old):
                    self._log(f"[DUP] {smarts} 与 {sma_old} 等价，未加入")
                    return False
            self._data[h].append((G, smarts))
            self._order.append(smarts)
            self._log(f"[ADD] 新增 SMARTS: {smarts}")
            return True

    def __contains__(self, smarts: str) -> bool:
        G = smarts_to_graph(smarts)
        if G is None:
            return False
        h = canonical_smarts_hash(smarts)
        if h not in self._data:
            return False
        for G_old, sma_old in self._data[h]:
            if are_graphs_isomorphic(G, G_old):
                return True
        return False

    def __len__(self):
        return len(self._order)

    def __iter__(self):
        return iter(self._order)

    def __repr__(self):
        return f"SmartsSet({self._order})"

    def get_logs(self):
        """导出日志记录"""
        return self._logs
