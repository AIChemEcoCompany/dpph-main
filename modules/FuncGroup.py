import json
import time
import pathlib
from copy import deepcopy
from loguru import logger
logger.remove(0)
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG, display
import matplotlib.pyplot as plt
import networkx as nx
from collections import namedtuple
from networkx.drawing.nx_agraph import graphviz_layout
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False

class Large_FG:
    def __init__(self, smiles = "", input_mol = None, smiles_with_atom_mapping=False):
        with open('data/substruct_functional-group.json', 'r', encoding='utf-8') as f:
            self.functional_group_substructure = json.load(f)
        
        self.functional_groups = sorted(list(set(self.functional_group_substructure.keys())))
        self.functional_groups_for_fg_search = deepcopy(self.functional_groups)
        self.smiles = smiles
        if input_mol is not None:
            self.mol = input_mol
        else:
            try:
                self.mol = Chem.MolFromSmiles(smiles)
            except:
                return None
        # 添加atom index    
        if not smiles_with_atom_mapping:
            try:
                for atom in self.mol.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx())
            except:
                logger.warning("Invalid SMILES")
                return None
        self.fg_dict = {}

    def _atom_mapping_to_index(self, am_in:int):
        atom_map = [atom.GetAtomMapNum() for atom in self.mol.GetAtoms()]
        try:
            return atom_map.index(am_in)
        except:
            logger.warning("atom mapping_in: {} is out of range for {} - {}".format(am_in, Chem.MolToSmiles(mol), atom_map))

    def _index_to_atom_mapping(self, index_in:int):
        atom_map = [atom.GetAtomMapNum() for atom in self.mol.GetAtoms()]
        try:
            return atom_map[index_in]
        except:
            logger.warning("index_in: {} is out of range for {} - {}".format(index_in, Chem.MolToSmiles(mol), atom_map))

    def find_FG(self, level=2):
        result_dict = {}

        if self.mol is None:
            logger.error("Invalid SMILES")
            return {"status": 400, "data": "Invalid SMILES"}

        for fg_name in self.functional_groups_for_fg_search:
            fg = self.functional_group_substructure[fg_name]
            fg_mol = Chem.MolFromSmarts(fg)
            match_result = Chem.Mol.GetSubstructMatches(self.mol, fg_mol, uniquify=True)
            if len(match_result) > 0:
                logger.info(fg_name)
                am_list = []
                for match in match_result:
                    am_list.append([self._index_to_atom_mapping(i) for i in match])
                logger.info(am_list)
                # logger.info(match_result)
                result_dict[fg_name] = am_list

        # 根据每个官能团包含的原子，排除已被更大官能团包含的官能团
        fg_dict_copy = self._exclude_small_fg(result_dict, level = level)

        logger.debug(fg_dict_copy)
        self.fg_dict = fg_dict_copy

        return {"status": 200, "data": fg_dict_copy}

    def _exclude_small_fg(self, fg_dict, level=2):
        fg_dict_copy = deepcopy(fg_dict)
        for fg_name_idx, fg_name in enumerate(fg_dict.keys()):
            sub_group_list = fg_dict[fg_name]
            # 遍历sub_group_list
            for sub_group_am_list_idx, sub_group_am_list in enumerate(sub_group_list):
                # 遍历所有其他官能团
                for other_fg_name_idx in range(fg_name_idx + 1, len(fg_dict.keys())):
                    other_fg_name = list(fg_dict.keys())[other_fg_name_idx]
                    other_sub_group_list = fg_dict[other_fg_name]
                    if fg_name == other_fg_name:
                        continue
                    for other_sub_group_am_list in other_sub_group_list:
                        # 如果sub_group_am_list包含other_sub_group_am_list，则删除other_fg_name
                        if set(other_sub_group_am_list).issubset(set(sub_group_am_list)):
                            if (set(other_sub_group_am_list) != set(sub_group_am_list)):
                                logger.debug("[RULE 1]Delete {} {} from fg_list".format(other_fg_name, other_sub_group_am_list))
                                # print("[RULE 1]Delete {} {} from fg_list".format(other_fg_name, other_sub_group_am_list))
                                try:
                                    fg_dict_copy[other_fg_name].remove(other_sub_group_am_list)
                                except Exception as e:
                                    continue
                            else:
                                # 检查两官能团模板之间的关系
                                this_fg_mol = Chem.MolFromSmarts(self.functional_group_substructure[fg_name])
                                other_fg_mol = Chem.MolFromSmarts(self.functional_group_substructure[other_fg_name])
                                this_has_other_flag = False
                                other_has_this_flag = False
                                try:
                                    if this_fg_mol.HasSubstructMatch(other_fg_mol):
                                        this_has_other_flag = True
                                except:
                                    pass
                                try:
                                    if other_fg_mol.HasSubstructMatch(this_fg_mol):
                                        other_has_this_flag = True
                                except:
                                    pass
                                if this_has_other_flag and not other_has_this_flag:
                                    logger.debug("[RULE 1]Delete {} {} from fg_list".format(other_fg_name, other_sub_group_am_list))
                                    # print("[RULE 1 - this_has_other_flag]Delete {} {} from fg_list".format(other_fg_name, other_sub_group_am_list))
                                    try:
                                        fg_dict_copy[other_fg_name].remove(other_sub_group_am_list)
                                    except Exception as e:
                                        continue
                                elif not this_has_other_flag and other_has_this_flag:
                                    logger.debug("[RULE 1]Delete {} {} from fg_list".format(fg_name, sub_group_am_list))
                                    # print("[RULE 1 - other_has_this_flag]Delete {} {} from fg_list".format(fg_name, sub_group_am_list))
                                    try:
                                        fg_dict_copy[fg_name].remove(sub_group_am_list)
                                    except Exception as e:
                                        continue
                                else:
                                    logger.debug("[RULE 1]Delete {} {} from fg_list".format(other_fg_name, other_sub_group_am_list))
                                    # print("[RULE 1 - FF]Delete {} {} from fg_list".format(other_fg_name, other_sub_group_am_list))
                                    try:
                                        fg_dict_copy[other_fg_name].remove(other_sub_group_am_list)
                                    except Exception as e:
                                        continue

        # 删除fg_dict_copy中value为空的键
        fg_dict_copy = {k: v for k, v in fg_dict_copy.items() if v}

        # 排除已被其他官能团包含的ethane
        if "ethane" in fg_dict_copy.keys():
            # 获取所有除"ethane"以外的键对应的值合并为一个list
            other_fg_list = [v for k, v in fg_dict_copy.items() if k != "ethane"]
            # 获取other_fg_list中所有list包含的元素
            other_fg_list = [item for sublist in other_fg_list for item in sublist]
            atoms_in_other_fg = set([item for sublist in other_fg_list for item in sublist])
            for sub_group_list in fg_dict_copy["ethane"]:
                if level == 2:
                    # 如果sub_group_list中的元素在other_fg_list中，则删除
                    # print("[RULE 2]Before delete ethane: {}".format(fg_dict_copy["ethane"]))
                    fg_dict_copy["ethane"] = [sub_group_list for sub_group_list in fg_dict_copy["ethane"] if len(set(sub_group_list).intersection(atoms_in_other_fg)) == 0]
                    # print("[RULE 2]After delete ethane: {}".format(fg_dict_copy["ethane"]))
                else:
                    fg_dict_copy["ethane"] = [sub_group_list for sub_group_list in fg_dict_copy["ethane"] if not set(sub_group_list).issubset(atoms_in_other_fg)]
            if len(fg_dict_copy["ethane"]) == 0:
                del fg_dict_copy["ethane"]

        if "carbonyl methylester" in fg_dict_copy.keys() and "ester" in fg_dict_copy.keys():
            for sub_group_list in fg_dict_copy["carbonyl methylester"]:
                for sub_group_list_ester in fg_dict_copy["ester"]:
                    if set(sub_group_list_ester) == set(sub_group_list):
                        fg_dict_copy["carbonyl methylester"].remove(sub_group_list)
                        logger.debug("[RULE ?]Delete carbonyl methylester: {}".format(sub_group_list))
                        break
            if len(fg_dict_copy["carbonyl methylester"]) == 0:
                del fg_dict_copy["carbonyl methylester"]
                    
        # 遍历原始数据结构，排除在其他所有value中list元素中出现的list元素
        if level in [1, 2]:
            filtered_data = {}
            for key_idx, key in enumerate(fg_dict_copy.keys()):
                value = fg_dict_copy[key]
                filtered_lists = []
                for idx, lst in enumerate(value):
                    # 获取其他所有官能团中包含的atom集合
                    all_other_value = set()
                    for other_key_idx in range(key_idx + 1, len(fg_dict_copy.keys())):
                        other_key = list(fg_dict_copy.keys())[other_key_idx]
                        other_value = fg_dict_copy[other_key]
                        for other_idx, other_lst in enumerate(other_value):
                            if key == other_key and idx == other_idx:
                                continue
                            else:
                                all_other_value.update(other_lst)
                    if set(lst).issubset(all_other_value):
                        logger.debug("[RULE 3] Delete {} {} from fg_list".format(key, lst))
                        # print("[RULE 3] Delete {} {} from fg_list".format(key, lst))
                        continue
                    else:
                        filtered_lists.append(lst)
                if len(filtered_lists) > 0:
                    filtered_data[key] = filtered_lists

            return filtered_data
        else:
            return fg_dict_copy

    def _visualize_sub_graph(self, sub_graph):
        plt.clf()
        # 可视化子图
        plt.figure(figsize=(6, 6))
        pos = graphviz_layout(sub_graph, prog='dot')
        nx.draw_networkx_nodes(sub_graph, pos, node_size=100, node_color='lightblue')
        nx.draw_networkx_edges(sub_graph, pos, alpha=0.7, width=0.5, edge_color='grey', style='dashed')
        nx.draw_networkx_labels(sub_graph, pos, font_size=8, font_color='black')
        plt.axis('off')
        pathlib.Path('img').mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        plt.savefig('img/sub_graph-{}.png'.format(now), dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_mol(self):
        # 创建一个绘图选项对象
        options = Draw.MolDrawOptions()
        # 设置为True，以便在可视化时显示原子索引
        options.includeAtomNumbers = True
        # 创建一个绘图对象，例如使用SVG绘制
        drawer = Draw.MolDraw2DSVG(600, 300)  # 设置画布大小为600x300像素
        drawer.SetDrawOptions(options)
        # 绘制分子
        drawer.DrawMolecule(self.mol)
        # 完成绘图
        drawer.FinishDrawing()
        # 获取SVG数据
        svg = drawer.GetDrawingText()
        # 显示SVG或者保存到文件
        display(SVG(svg))
        pathlib.Path('img').mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        with open("img/{}.svg".format(now), "w") as f:
            f.write(svg)

    def visualize_mol_with_highlight_each_fg(self, fg):
        for idx, (func_group, atom_lists) in enumerate(fg.items()):
            # 使用matplotlib的colormap生成颜色
            colormap = plt.cm.get_cmap('Pastel1')
            def get_color_map(num_colors):
                return [colormap(i / num_colors) for i in range(num_colors)]
            # 为这个分子的官能团生成颜色映射
            colors = get_color_map(1)
            highlight_atoms = []
            highlight_bonds = []
            atom_colors = {}
            bond_colors = {}

            # 创建一个绘图选项对象
            options = Draw.MolDrawOptions()
            # 设置为True，以便在可视化时显示原子索引
            options.includeAtomNumbers = True
            # 根据官能团的原子高亮显示
            # TODO: 这里多次重复调用_atom_mapping_to_index，可优化以提升速度
            color = colors[0]
            for atom_list in atom_lists:
                if isinstance(atom_list, list):
                    highlight_atoms.extend([self._atom_mapping_to_index(atom) for atom in atom_list])
                    for atom in atom_list:
                        atom_colors[self._atom_mapping_to_index(atom)] = color
                    for i in range(len(atom_list) - 1):
                        for j in range(i + 1, len(atom_list)):
                            try:
                                bond = self.mol.GetBondBetweenAtoms(self._atom_mapping_to_index(atom_list[i]), self._atom_mapping_to_index(atom_list[j]))
                            except:
                                bond = None
                            if bond:
                                highlight_bonds.append(bond.GetIdx())
                                bond_colors[bond.GetIdx()] = color
                else:
                    highlight_atoms.append(self._atom_mapping_to_index(atom_list))
                    atom_colors[self._atom_mapping_to_index(atom_list)] = color
                
            # 创建一个绘图对象，例如使用SVG绘制
            drawer = Draw.MolDraw2DSVG(600, 300)  # 设置画布大小为600x300像素
            drawer.SetDrawOptions(options)
            # 绘制分子
            drawer.DrawMolecule(self.mol,
                                highlightAtoms=highlight_atoms,
                                highlightBonds=highlight_bonds,
                                highlightAtomColors=atom_colors,
                                highlightBondColors=bond_colors)
            # 完成绘图
            drawer.FinishDrawing()
            # 获取SVG数据
            svg = drawer.GetDrawingText()
            # 显示SVG或者保存到文件
            print("{}: ".format(func_group))
            display(SVG(svg))

    def visualize_mol_with_highlight(self, fg):
        # 使用matplotlib的colormap生成颜色
        colormap = plt.cm.get_cmap('Pastel1')
        def get_color_map(num_colors):
            return [colormap(i / num_colors) for i in range(num_colors)]
        # 为这个分子的官能团生成颜色映射
        colors = get_color_map(len(fg))
        highlight_atoms = []
        highlight_bonds = []
        atom_colors = {}
        bond_colors = {}

        # 创建一个绘图选项对象
        options = Draw.MolDrawOptions()
        # 设置为True，以便在可视化时显示原子索引
        options.includeAtomNumbers = True
        # 根据官能团的原子高亮显示
        # TODO: 这里多次重复调用_atom_mapping_to_index，可优化以提升速度
        for idx, (func_group, atom_lists) in enumerate(fg.items()):
            color = colors[idx]
            for atom_list in atom_lists:
                if isinstance(atom_list, list):
                    highlight_atoms.extend([self._atom_mapping_to_index(atom) for atom in atom_list])
                    for atom in atom_list:
                        atom_colors[self._atom_mapping_to_index(atom)] = color
                    for i in range(len(atom_list) - 1):
                        for j in range(i + 1, len(atom_list)):
                            try:
                                bond = self.mol.GetBondBetweenAtoms(self._atom_mapping_to_index(atom_list[i]), self._atom_mapping_to_index(atom_list[j]))
                            except:
                                bond = None
                            if bond:
                                highlight_bonds.append(bond.GetIdx())
                                bond_colors[bond.GetIdx()] = color
                else:
                    highlight_atoms.append(self._atom_mapping_to_index(atom_list))
                    atom_colors[self._atom_mapping_to_index(atom_list)] = color
            
        # 创建一个绘图对象，例如使用SVG绘制
        drawer = Draw.MolDraw2DSVG(600, 300)  # 设置画布大小为600x300像素
        drawer.SetDrawOptions(options)
        # 绘制分子
        drawer.DrawMolecule(self.mol,
                            highlightAtoms=highlight_atoms,
                            highlightBonds=highlight_bonds,
                            highlightAtomColors=atom_colors,
                            highlightBondColors=bond_colors)
        # 完成绘图
        drawer.FinishDrawing()
        # 获取SVG数据
        svg = drawer.GetDrawingText()
        # 显示SVG或者保存到文件
        display(SVG(svg))
        return svg

class FuncGroups:
    def __init__(
        self,
    ):
        # List obtained from https://github.com/rdkit/rdkit/blob/master/Data/FunctionalGroups.txt
        self.dict_fgs = {
            "furan": "o1cccc1",
            "aldehydes": " [CX3H1](=O)[#6]",
            "esters": " [#6][CX3](=O)[OX2H0][#6]",
            "ketones": " [#6][CX3](=O)[#6]",
            "amides": " C(=O)-N",
            "thiol groups": " [SH]",
            "alcohol groups": " [OH]",
            "methylamide": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
            "carboxylic acids": "*-C(=O)[O;D1]",
            "carbonyl methylester": "*-C(=O)[O;D2]-[C;D1;H3]",
            "terminal aldehyde": "*-C(=O)-[C;D1]",
            "amide": "*-C(=O)-[N;D1]",
            "carbonyl methyl": "*-C(=O)-[C;D1;H3]",
            "isocyanate": "*-[N;D2]=[C;D2]=[O;D1]",
            "isothiocyanate": "*-[N;D2]=[C;D2]=[S;D1]",
            "nitro": "*-[N;D3](=[O;D1])[O;D1]",
            "nitroso": "*-[N;R0]=[O;D1]",
            "oximes": "*=[N;R0]-[O;D1]",
            "Imines": "*-[N;R0]=[C;D1;H2]",
            "terminal azo": "*-[N;D2]=[N;D2]-[C;D1;H3]",
            "hydrazines": "*-[N;D2]=[N;D1]",
            "diazo": "*-[N;D2]#[N;D1]",
            "cyano": "*-[C;D2]#[N;D1]",
            "primary sulfonamide": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
            "methyl sulfonamide": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
            "sulfonic acid": "*-[S;D4](=O)(=O)-[O;D1]",
            "methyl ester sulfonyl": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
            "methyl sulfonyl": "*-[S;D4](=O)(=O)-[C;D1;H3]",
            "sulfonyl chloride": "*-[S;D4](=O)(=O)-[Cl]",
            "methyl sulfinyl": "*-[S;D3](=O)-[C;D1]",
            "methyl thio": "*-[S;D2]-[C;D1;H3]",
            "thiols": "*-[S;D1]",
            "thio carbonyls": "*=[S;D1]",
            "halogens": "*-[#9,#17,#35,#53]",
            "t-butyl": "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
            "tri fluoromethyl": "*-[C;D4](F)(F)F",
            "acetylenes": "*-[C;D2]#[C;D1;H]",
            "cyclopropyl": "*-[C;D3]1-[C;D2]-[C;D2]1",
            "ethoxy": "*-[O;D2]-[C;D2]-[C;D1;H3]",
            "methoxy": "*-[O;D2]-[C;D1;H3]",
            "side-chain hydroxyls": "*-[O;D1]",
            "ketones": "*=[O;D1]",
            "primary amines": "*-[N;D1]",
            "nitriles": "*#[N;D1]",
        }

    def _is_fg_in_mol(self, mol, fg):
        fgmol = Chem.MolFromSmarts(fg)
        mol = Chem.MolFromSmiles(mol.strip())
        return len(Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True)) > 0

    def check(self, smiles: str) -> str:
        """
        Input a molecule SMILES or name.
        Returns a list of functional groups identified by their common name (in natural language).
        """
        try:
            fgs_in_molec = [
                (name, fg)
                for name, fg in self.dict_fgs.items()
                if self._is_fg_in_mol(smiles, fg)
            ]
            print("[RDKit 识别结果：]")
            for name, fg in fgs_in_molec[:-1]:
                print(f"{name}: {fg}")
        except:
            print("Wrong argument. Please input a valid molecular SMILES.")

# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)
    
def merge(mol, marked, aset):
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)
    
def identify_functional_groups(mol):
    marked = set()
    #mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
            marked.add(atom.GetIdx())

    #mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

    #merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)


    #extract also connected unmarked carbon atoms
    ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
    ifgs = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        ifgs.append(
            ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
                type=Chem.MolFragmentToSmiles(mol, g.union(uca), canonical=True)))
    return ifgs
