# %%
import requests
import json
import time
import numpy as np

from modules.atom_mapping import data_process
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from modules.FuncGroup import Large_FG

# %%
class RXN:
    """
    Represents a chemical reaction.

    Args:
        reactants (list): List of SMILES strings representing the reactants.
        products (list): List of SMILES strings representing the products.
        am (bool, optional): Flag indicating whether to perform atom mapping. Defaults to True.

    Attributes:
        reactants (list): List of dictionaries representing the reactants, with each dictionary containing the SMILES string.
        products (list): List of dictionaries representing the products, with each dictionary containing the SMILES string.
        rxn_smiles (str): SMILES string representation of the reaction.
        rxn_obj (rdChemReactions.ChemicalReaction): RDKit ChemicalReaction object representing the reaction.
        reactant_bond_type (dict): Dictionary mapping bond tuples to bond types for the reactants.
        product_bond_type (dict): Dictionary mapping bond tuples to bond types for the products.
        symbols (dict): Dictionary mapping atom map numbers to atom symbols.
        Hs_reactants (dict): Dictionary mapping atom map numbers to the total number of hydrogens for the reactants.
        Hs_products (dict): Dictionary mapping atom map numbers to the total number of hydrogens for the products.
        reactant_rings (list): List of lists representing the rings in the reactants, with each inner list containing the atom map numbers of the atoms in the ring.
        product_rings (list): List of lists representing the rings in the products, with each inner list containing the atom map numbers of the atoms in the ring.
        atom_in_reactant_rings (list): List of atom map numbers of atoms in the reactant rings.
        atom_in_product_rings (list): List of atom map numbers of atoms in the product rings.

    Methods:
        get_bond_change(): Returns the broken and formed bonds in the reaction.

    """

    def __init__(self, reactants, products, am=True):
        # self.API = "http://101.33.241.212:8757/mapping_smiles"
        if am:
            self.reactants, self.products = self._get_atom_mapping(reactants, products)
        else:
            self.reactants = [{"SMILES": x} for x in reactants]
            self.products = [{"SMILES": x} for x in products]
        # 根据底物个数初始化底物断键list
        self.broken_each_reactant = [[] for _ in range(len(self.reactants))]
        # 根据产物个数初始化产物成键list
        self.formed_each_product = [[] for _ in range(len(self.products))]
        self.rxn_smiles = ".".join([x["SMILES"] for x in self.reactants]) + ">>" + ".".join([x["SMILES"] for x in self.products])
        self.rxn_obj = rdChemReactions.ReactionFromSmarts(self.rxn_smiles, useSmiles=True)
        self._get_global_adjacency_matrix()
        # self.reactant_fg, self.product_fg = self._check_functional_groups()                 # 底物产物中包含的官能团
        # 识别底物产物中所有的环结构，及其中包含的原子的atom mapping
        self.reactant_bond_type, self.product_bond_type = self._get_bond_type()
        self.symbols, self.Hs_reactants, self.Hs_products = self._get_symbols_and_Hs()
        self.atom_in_reactants, self.atom_in_all_reactants = self._get_atom_in_reactants()
        self.atom_in_products, self.atom_in_all_products = self._get_atom_in_products()
        # 识别底物产物中的环结构
        # self.reactant_rings, self.product_rings, self.atom_in_reactant_rings, self.atom_in_product_rings = self._get_rings()

    def _atom_mapping_func(self, reaction_list):
        start = time.time()
        result = data_process(reaction_list)[0]
        end = time.time()
        logger.debug("调用Atom mapping function耗时： %.2f s"%(end-start))
        return result

    def _atom_mapping_API(self, reaction_list):
        message = json.dumps(
            {
                "reactions" : reaction_list
            }
        )
        headers = {'content-type': 'application/json'}
        start = time.time()
        res = requests.post(self.API, data=message,headers=headers)
        end = time.time()
        logger.debug("调用Atom mapping API耗时： %.2f s"%(end-start))
        return res.json()["data"][0]

    def _get_atom_mapping(self, reactants, products):
        r_list = []
        for r_smiles in reactants:
            r_list.append({
                'role': 'substrate',
                'smiles': r_smiles,
            })
        for p_smiles in products:
            r_list.append({
                'role': 'product',
                'smiles': p_smiles,
            })

        reaction_list = [r_list]
        # result = self._atom_mapping_API(reaction_list)
        result = self._atom_mapping_func(reaction_list)

        result_reactants = []
        result_products = []
        for item in result:
            if "role" in item.keys():
                if "mapped_smiles" not in item.keys():
                    raise ValueError("Atom mapping failed")
                else:
                    if item["role"] == "substrate":
                        result_reactants.append({"SMILES": item["mapped_smiles"]})
                    if item["role"] == "product":
                        result_products.append({"SMILES": item["mapped_smiles"]})
        return result_reactants, result_products
    
    def _index_to_atom_mapping(self, mol, index_in:int):
        atom_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        try:
            return atom_map[index_in]
        except:
            logger.warning("index_in: {} is out of range for {} - {}".format(index_in, Chem.MolToSmiles(mol), atom_map))

    def _get_global_adjacency_matrix(self):
        # 获取所有反应物的atom mapping的最小值和最大值
        atom_mapping_max = 0
        for reactant_mol in self.rxn_obj.GetReactants():
            atom_mapping = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
            atom_mapping_max = max(atom_mapping_max, max(atom_mapping))
        for product in self.rxn_obj.GetProducts():
            matrix = Chem.GetAdjacencyMatrix(product)
            atom_mapping = [atom.GetAtomMapNum() for atom in product.GetAtoms()]
            atom_mapping_max = max(atom_mapping_max, max(atom_mapping))

        # 构造全局邻接矩阵
        for idx, reactant_mol in enumerate(self.rxn_obj.GetReactants()):
            matrix = Chem.GetAdjacencyMatrix(reactant_mol)
            global_matrix = np.zeros((atom_mapping_max+1, atom_mapping_max+1), dtype=int)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i][j] == 1:
                        global_matrix[self._index_to_atom_mapping(reactant_mol, i)][self._index_to_atom_mapping(reactant_mol, j)] = 1
            self.reactants[idx]["mol"] = reactant_mol
            self.reactants[idx]["global_matrix"] = global_matrix
            # logger.debug("{} 的邻接矩阵：\n{}".format(self.reactants[idx]["SMILES"], global_matrix))
        
        for idx, product_mol in enumerate(self.rxn_obj.GetProducts()):
            matrix = Chem.GetAdjacencyMatrix(product_mol)
            global_matrix = np.zeros((atom_mapping_max+1, atom_mapping_max+1), dtype=int)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i][j] == 1:
                        global_matrix[self._index_to_atom_mapping(product_mol, i)][self._index_to_atom_mapping(product_mol, j)] = 1
            self.products[idx]["mol"] = product_mol
            self.products[idx]["global_matrix"] = global_matrix
            # logger.debug("{} 的邻接矩阵：\n{}".format(self.products[idx]["SMILES"], global_matrix))
        logger.debug("全局邻接矩阵构造完成")

    def _get_atom_in_reactants(self):
        atom_in_reactants = []
        for reactant in self.reactants:
            atom_in_reactants.append(sorted([atom.GetAtomMapNum() for atom in reactant["mol"].GetAtoms()]))
        # 创建一个新的变量，获取所有底物的atom mapping
        atom_in_all_reactants = [item for sublist in atom_in_reactants for item in sublist]        
        # 对atom_in_reactants去重并排序
        atom_in_all_reactants = list(set(atom_in_all_reactants))
        atom_in_all_reactants.sort()
        return atom_in_reactants, atom_in_all_reactants

    def _get_atom_in_products(self):
        atom_in_products = []
        for product in self.products:
            atom_in_products.append(sorted([atom.GetAtomMapNum() for atom in product["mol"].GetAtoms()]))
        # 创建一个新的变量，获取所有产物的atom mapping
        atom_in_all_products = [item for sublist in atom_in_products for item in sublist]        
        # 对atom_in_products去重并排序
        atom_in_all_products = list(set(atom_in_all_products))
        atom_in_all_products.sort()
        return atom_in_products, atom_in_all_products

    # 识别底物产物中所有的环结构，及其中包含的原子的atom mapping
    def _get_rings(self):
        atom_in_r_rings = []
        atom_in_p_rings = []
        reactant_rings = []
        product_rings = []
        for item in self.reactants:
            this_r_rings = []
            rings = item["mol"].GetRingInfo().AtomRings()
            if len(rings) > 0:
                # 将rings中的原子的atom mapping存储到reactant_rings中
                for ring in rings:
                    this_ring = sorted([self._index_to_atom_mapping(item["mol"], atom) for atom in ring])
                    this_r_rings.append(this_ring)
                    atom_in_r_rings += this_ring
                reactant_rings += this_r_rings
        for item in self.products:
            this_p_rings = []
            rings = item["mol"].GetRingInfo().AtomRings()
            if len(rings) > 0:
                # 将rings中的原子的atom mapping存储到product_rings中
                for ring in rings:
                    this_ring = sorted([self._index_to_atom_mapping(item["mol"], atom) for atom in ring])
                    this_p_rings.append(this_ring)
                    atom_in_p_rings += this_ring
                product_rings += this_p_rings
        
        return reactant_rings, product_rings, atom_in_r_rings, atom_in_p_rings

    # 定义函数识别底物产物中包含的官能团:
    def _check_functional_groups(self):
        reactant_fg = []
        product_fg = []
        for item in self.reactants:
            reactant = item["SMILES"]
            large_fg = Large_FG(reactant, smiles_with_atom_mapping=False)
            result_dict = large_fg.find_FG()["data"]
            reactant_fg.append([key for key in result_dict.keys()])
        for item in self.products:
            product = item["SMILES"]
            large_fg = Large_FG(product, smiles_with_atom_mapping=False)
            result_dict = large_fg.find_FG()["data"]
            product_fg.append([key for key in result_dict.keys()])
        return reactant_fg, product_fg

    def _get_bond_type(self):
        reactant_bond_type = {}
        product_bond_type = {}
        for item in self.reactants:
            for bond in item["mol"].GetBonds():
                reactant_bond_type[(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())] = str(bond.GetBondType())
                reactant_bond_type[(bond.GetEndAtom().GetAtomMapNum(), bond.GetBeginAtom().GetAtomMapNum())] = str(bond.GetBondType())
        for item in self.products:
            for bond in item["mol"].GetBonds():
                product_bond_type[(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())] = str(bond.GetBondType())
                product_bond_type[(bond.GetEndAtom().GetAtomMapNum(), bond.GetBeginAtom().GetAtomMapNum())] = str(bond.GetBondType())
        return reactant_bond_type, product_bond_type

    def _get_symbols_and_Hs(self):
        symbols = {}
        Hs_reactants = {}
        Hs_products = {}
        for item in self.reactants:
            Chem.SanitizeMol(item["mol"])
            for atom in item["mol"].GetAtoms():
                symbols[atom.GetAtomMapNum()] = atom.GetSymbol()
                Hs_reactants[atom.GetAtomMapNum()] = atom.GetTotalNumHs()
        for item in self.products:
            Chem.SanitizeMol(item["mol"])
            for atom in item["mol"].GetAtoms():
                symbols[atom.GetAtomMapNum()] = atom.GetSymbol()
                Hs_products[atom.GetAtomMapNum()] = atom.GetTotalNumHs()
        return symbols, Hs_reactants, Hs_products

    def _check_in_which_r(self, i, j=None):
        for idx, atom_list in enumerate(self.atom_in_reactants):
            if j is None:
                if i in atom_list:
                    return idx
            else:
                if i in atom_list and j in atom_list:
                    return idx
        return None

    def _check_in_which_p(self, i, j=None):
        for idx, atom_list in enumerate(self.atom_in_products):
            if j is None:
                if i in atom_list:
                    return idx
            else:
                if i in atom_list and j in atom_list:
                    return idx
        return None

    def get_bond_change(self, with_idx=False, consider_broken_inadequacy = False):
        """
        Returns the broken and formed bonds in the reaction.

        Returns:
            tuple: A tuple containing three lists:
                - broken (list): List of strings representing the broken bonds in the reaction.
                - formed (list): List of strings representing the formed bonds in the reaction.
                - bond_changed (list): List of strings representing the bonds whose type has changed in the reaction.
        """

        def construct_broken_formed_bonds(self, i, j, broken=True, with_idx=False):
            """
            Constructs the string representation of a broken bond.

            Args:
                i (int): Index of the first atom in the bond.
                j (int): Index of the second atom in the bond.

            Returns:
                str: String representation of the broken bond.
            """
            if self.symbols[i] <= self.symbols[j]:
                if with_idx:
                    bond_str_list = [self.symbols[i] + ":" + str(i), "", self.symbols[j] + ":" + str(j)]
                else:
                    bond_str_list = [self.symbols[i], "", self.symbols[j]]
            else:
                if with_idx:
                    bond_str_list = [self.symbols[j] + ":" + str(j), "", self.symbols[i] + ":" + str(i)]
                else:
                    bond_str_list = [self.symbols[j], "", self.symbols[i]]
            # Construct the bond string based on the bond type
            if broken:
                if self.reactant_bond_type[(i, j)] == "SINGLE":
                    bond_str_list[1] = " - "
                elif self.reactant_bond_type[(i, j)] == "DOUBLE":
                    bond_str_list[1] = " = "
                elif self.reactant_bond_type[(i, j)] == "TRIPLE":
                    bond_str_list[1] = " # "
                else:
                    # Convert to lowercase  aromatic
                    bond_str_list[1] = " " + self.reactant_bond_type[(i, j)].lower() + " "
            else:
                if self.product_bond_type[(i, j)] == "SINGLE":
                    bond_str_list[1] = " - "
                elif self.product_bond_type[(i, j)] == "DOUBLE":
                    bond_str_list[1] = " = "
                elif self.product_bond_type[(i, j)] == "TRIPLE":
                    bond_str_list[1] = " # "
                else:
                    # Convert to lowercase
                    bond_str_list[1] = " " + self.product_bond_type[(i, j)].lower() + " "
            bond_str = "".join(bond_str_list)
            # Check which reactant the bond belongs to
            if broken:
                idx = self._check_in_which_r(i, j)
            else:
                idx = self._check_in_which_p(i, j)
            if idx is not None:
                if broken:
                    self.broken_each_reactant[idx].append(bond_str)
                else:
                    self.formed_each_product[idx].append(bond_str)
            return bond_str

        broken = []
        formed = []
        bond_changed = []

        # 对所有产物的global_matrix相加
        matrix = np.zeros(self.products[0]["global_matrix"].shape, dtype=int)
        for product in self.products:
            # matrix = matrix + product["global_matrix"]
            result = (matrix != 0) | (product["global_matrix"] != 0)
            matrix = result.astype(int)
        r_matrix = np.zeros(self.products[0]["global_matrix"].shape, dtype=int)
        for reactant in self.reactants:
            result = (r_matrix != 0) | (reactant["global_matrix"] != 0)
            r_matrix = result.astype(int)
        matrix = matrix - r_matrix
        # 从matrix中找出值为-1的元素的坐标
        # display(self.rxn_obj)
        for i in range(matrix.shape[0]):
            for j in range(i+1, matrix.shape[1]):
                if matrix[i][j] == -1:
                    # 如果断键的两端原子均未在产物中出现，则不记录
                    if (i not in self.atom_in_all_products) and (j not in self.atom_in_all_products):
                        logger.debug("Atoms in broken bond {} {} - {} {} not in products".format(i, self.symbols[i], j, self.symbols[j]))
                        continue
                    else:
                        logger.debug("Broken: {} {} - {} {} {} bond".format(i, self.symbols[i], j, self.symbols[j], self.reactant_bond_type[(i, j)]))
                        broken.append(construct_broken_formed_bonds(self, i, j, with_idx=with_idx))
                elif matrix[i][j] == 1:
                    # 如果形成的键的两端原子均未在底物中出现，则不记录
                    if (i not in self.atom_in_all_reactants) and (j not in self.atom_in_all_reactants):
                        logger.debug("Atoms in formed bond {} {} - {} {} not in reactants".format(i, self.symbols[i], j, self.symbols[j]))
                        continue
                    else:
                        logger.debug("Formed: {} {} - {} {} {} bond".format(i, self.symbols[i], j, self.symbols[j], self.product_bond_type[(i, j)]))
                        formed.append(construct_broken_formed_bonds(self, i, j, broken=False, with_idx=with_idx))
                else:
                    if (i, j) in self.reactant_bond_type.keys() and (i, j) in self.product_bond_type.keys() and self.reactant_bond_type[(i, j)] != self.product_bond_type[(i, j)]:
                        # 键类型改变
                        logger.debug("Changed: {} {} - {} {} {} bond to {} bond".format(i, self.symbols[i], j, self.symbols[j], self.reactant_bond_type[(i, j)], self.product_bond_type[(i, j)]))
                        
                        if self.symbols[i] <= self.symbols[j]:
                            if with_idx:
                                bond_str_list = [self.symbols[i] + ":" + str(i), "", self.symbols[j] + ":" + str(j), " -> ", self.symbols[i] + ":" + str(i), "", self.symbols[j] + ":" + str(j)]
                            else:
                                bond_str_list = [self.symbols[i], "", self.symbols[j], " -> ", self.symbols[i], "", self.symbols[j]]
                        else:
                            if with_idx:
                                bond_str_list = [self.symbols[j] + ":" + str(j), "", self.symbols[i] + ":" + str(i), " -> ", self.symbols[j] + ":" + str(j), "", self.symbols[i] + ":" + str(i)]
                            else:
                                bond_str_list = [self.symbols[j], "", self.symbols[i], " -> ", self.symbols[j], "", self.symbols[i]]
                        # Construct the bond string based on the bond type
                        if self.reactant_bond_type[(i, j)] == "SINGLE":
                            bond_str_list[1] = " - "
                        elif self.reactant_bond_type[(i, j)] == "DOUBLE":
                            bond_str_list[1] = " = "
                        elif self.reactant_bond_type[(i, j)] == "TRIPLE":
                            bond_str_list[1] = " # "
                        else:
                            # Convert to lowercase
                            bond_str_list[1] = " " + self.reactant_bond_type[(i, j)].lower() + " "
                        
                        if self.product_bond_type[(i, j)] == "SINGLE":
                            bond_str_list[5] = " - "
                        elif self.product_bond_type[(i, j)] == "DOUBLE":
                            bond_str_list[5] = " = "
                        elif self.product_bond_type[(i, j)] == "TRIPLE":
                            bond_str_list[5] = " # "
                        else:
                            # Convert to lowercase
                            bond_str_list[5] = " " + self.product_bond_type[(i, j)].lower() + " "
                        
                        bond_str = "".join(bond_str_list)
                        
                        # 检查i j 属于哪个底物
                        idx = self._check_in_which_r(i, j)
                        if idx is not None and '->' not in bond_str :#添加2025-2-11 排除键改变的
                            self.broken_each_reactant[idx].append(bond_str)
                        
                        ################################################################### 添加弱断键类型 键类型改变
                        elif consider_broken_inadequacy and idx is not None and '->' in bond_str :
                            self.broken_each_reactant[idx].append(bond_str.split("->")[0])
                            idx_ = self._check_in_which_p(i, j)
                            if idx_ is not None:
                                self.formed_each_product[idx_].append(bond_str.split("->")[1])
                        bond_changed.append(bond_str)
        # 检查每个原子反应前后的Hs是否相同
        for key in self.Hs_products.keys(): #{"atom idx": Hs}
            try:
                if (key not in self.Hs_reactants.keys()) and (key in self.Hs_products.keys()):
                    logger.debug("Formed: {} {} - H bond".format(key, self.symbols[key]))
                    if key in self.atom_in_all_reactants:
                        # 检查key 属于哪个产物
                        idx = self._check_in_which_p(i=key)
                        # 检查原子是否在底物中出现
                        if idx is not None:
                            if with_idx:
                                self.formed_each_product[idx].append("{}.{} - H".format(key, self.symbols[key]))
                            else:
                                self.formed_each_product[idx].append("{} - H".format(self.symbols[key]))
                        if with_idx:
                            formed.append("{}:{} - H".format(self.symbols[key], key))
                        else:
                            formed.append("{} - H".format(self.symbols[key]))
                elif (key in self.Hs_reactants.keys()) and (key not in self.Hs_products.keys()):
                    logger.debug("Broken: {} {} - H bond".format(key, self.symbols[key]))
                    if key in self.atom_in_all_products:
                        # 检查key 属于哪个底物
                        idx = self._check_in_which_r(i=key)
                        if idx is not None:
                            if with_idx:
                                self.broken_each_reactant[idx].append("{}.{} - H".format(key, self.symbols[key]))
                            else:
                                self.broken_each_reactant[idx].append("{} - H".format(self.symbols[key]))
                        if with_idx:
                            broken.append("{}:{} - H".format(self.symbols[key], key))
                        else:
                            broken.append("{} - H".format(self.symbols[key]))
                else:
                    if self.Hs_reactants[key] > self.Hs_products[key]:
                        if key in self.atom_in_all_products:
                            logger.debug("Broken: {} {} - H bond".format(key, self.symbols[key]))
                            # 检查key 属于哪个底物
                            idx = self._check_in_which_r(i=key)
                            if idx is not None:
                                if with_idx:
                                    self.broken_each_reactant[idx].append("{}:{} - H".format(self.symbols[key], key))
                                else:
                                    self.broken_each_reactant[idx].append("{} - H".format(self.symbols[key]))
                            if with_idx:
                                broken.append("{}:{} - H".format(self.symbols[key], key))
                            else:
                                broken.append("{} - H".format(self.symbols[key]))
                    elif self.Hs_reactants[key] < self.Hs_products[key]:
                        logger.debug("Formed: {} {} - H bond".format(key, self.symbols[key]))
                        if key in self.atom_in_all_reactants:
                            # 检查key 属于哪个产物
                            idx = self._check_in_which_p(i=key)
                            if idx is not None:
                                if with_idx:
                                    self.formed_each_product[idx].append("{}:{} - H".format(self.symbols[key], key))
                                else:
                                    self.formed_each_product[idx].append("{} - H".format(self.symbols[key]))
                            if with_idx:
                                formed.append("{}:{} - H".format(self.symbols[key], key))
                            else:
                                formed.append("{} - H".format(self.symbols[key]))
            except:
                logger.error("{} - {} - {} - {}".format(self.reactants[0]["SMILES"], self.reactants[1]["SMILES"], self.products[0]["SMILES"], key))
                raise ValueError("Hs_reactants and Hs_products are not the same length")
        
        try:
            if consider_broken_inadequacy: # DOUBLE -> SINGLE    TRIPLE -> DOUBLE
                borken_, formed_ = set(broken), set(formed)
                for b in bond_changed:
                    borken_.add(b.split("->")[0])
                    formed_.add(b.split("->")[1])

                broken, formed = list(borken_), list(formed_)
                # print(broken, formed, bond_changed)
                return broken, formed, bond_changed
        except:
            raise KeyError()
        return broken, formed, bond_changed
    
    
    def _get_which_r(self):
        for idx, atom_list in enumerate(self.atom_in_reactants):
            mol = Chem.MolFromSmiles(atom_list)
            
            for atom in mol.GetAtoms():
                m = atom.GetAtomMapNum()
    
    def get_bond_change_for_each(self):
        return self.broken_each_reactant, self.formed_each_product

    def get_fg_for_each(self):
        return self.reactant_fg, self.product_fg

# %%
# from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.molSize = (600,250)
# IPythonConsole.highlightByReactant = True

# # sub1, sub2, prod = ('O[C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][c:10]([Cl:11])[c:12]1[NH2:13]', 'Cc1c(Br)cc(CC(=O)O)cc1', '[O:1]=[c:2]1[nH:3][c:4](=[O:5])[c:6]2[cH:7][cH:8][cH:9][c:10]([Cl:11])[c:12]2[nH:13]1')
# sub1, prod = ('CS(C)(=O)=CC(=O)C1=CC=CC=C1', 'O=C1OC(=CC2=CC=CC=C12)C1=CC=CC=C1')

# # sub1, sub2, prod = ('[CH3:1][O:2][C:3](=[O:4])[CH:5]([C:6]#[N:7])[CH2:8][C:9]1=[N:22][O:21][C:18]([CH3:19])([CH3:20])[CH2:17][CH:10]1[c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1', 'Br[C:23]([CH3:24])=[O:25]', '[CH3:1][O:2][C:3](=[O:4])[CH:5]([C:6]#[N:7])[CH2:8][C:9]1=[C:10]([c:11]2[cH:12][cH:13][cH:14][cH:15][cH:16]2)[CH2:17][C:18]([CH3:19])([CH3:20])[O:21][N:22]1[C:23]([CH3:24])=[O:25]')
# rxn = RXN([sub1], [prod], am=True)
# rxn.get_bond_change()
# reactants, products = rxn.rxn_obj.GetReactants(), rxn.rxn_obj.GetProducts()
# display(reactants[0]), display(products[0])

# %%