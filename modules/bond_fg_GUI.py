import time
import json
import pickle
from tqdm import tqdm
import pandas as pd
from loguru import logger
logger.remove(0)
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython.display import SVG


from localmapper import localmapper

from chem_balancer.main import masterbalance
from modules.atom_mapping import data_process
from modules.rxn4bond import RXN
from modules.FuncGroup import Large_FG, FuncGroups
from modules.utils import draw_chemical_reaction

class Data_Obj:
    #化学反应式子的相关信息（底物 产物 条件）
    def __init__(
            self,
            idx,
            row,
            sub_col_name_list = ["底物1", "底物2", "底物3", "底物4", "底物5", "底物6"],
            prod_col_name_list = ["产物1", "产物2", "产物3", "产物4", "产物5", "产物6", "产物7"],
            reagent_col_name_list = ["Reagent1", "Reagent2", "Reagent3", "Reagent4", "Reagent5", "Reagent6", "Reagent7",
                                     "Reagent8", "Reagent9", "Reagent10", "Reagent11", "Reagent12", "Reagent13", "Reagent14",
                                     "Reagent15", "Reagent16", "Reagent17", "Reagent18", "Reagent19", "Reagent20", "Reagent21"],
            solvent_col_name_list = ["Solvent1", "Solvent2", "Solvent3", "Solvent4", "Solvent5", "Solvent6", "Solvent7",
                                     "Solvent8", "Solvent9", "Solvent10"],
            cond_id_col_name = "条件编号"
        ):
        self.sub_col_name_list = sub_col_name_list
        self.prod_col_name_list = prod_col_name_list
        self.reagent_col_name_list = reagent_col_name_list
        self.solvent_col_name_list = solvent_col_name_list
        self.condition_id_col_name = cond_id_col_name

        self.row = row
        self.row_index = idx
        self.fg_each_reactant_list = []
        self.fg_each_product_list = []

    def _get_atom_mapping(self, row, localmapper):
        r_smiles_list = []
        p_smiles_list = []

        # 在做atom mapping前先检查底物分子量是否小于产物的分子量，如果是，则先尝试配平
        r_mw = 0
        p_mw = 0
        for col_name in self.sub_col_name_list:
            # 获取row[col_name]的全小写
            if row[col_name] is None or pd.isna(row[col_name]) or row[col_name] == "" or row[col_name] == " " or row[col_name].lower() == "nan":
                continue
            else:
                r_smiles_list.append(row[col_name].strip())
                try:
                    r_mol = Chem.MolFromSmiles(row[col_name])
                    r_mw += Descriptors.MolWt(r_mol)
                except Exception as e:
                    logger.error(e)
                    logger.error(row[col_name])
                    print("\n【获取底物分子量失败】")
                    return False, False, [], []
        for col_name in self.prod_col_name_list:
            if row[col_name] is None or pd.isna(row[col_name]) or row[col_name] == "" or row[col_name] == " " or row[col_name].lower() == "nan":
                continue
            else:
                p_smiles_list.append(row[col_name].strip())
                try:
                    p_mol = Chem.MolFromSmiles(row[col_name])
                    p_mw += Descriptors.MolWt(p_mol)
                except Exception as e:
                    logger.error(e)
                    logger.error(row[col_name])
                    print("\n【获取产物分子量失败】")
                    return False, False, [], []

        to_be_balanced = False
        if len(r_smiles_list) == 0 or len(p_smiles_list) == 0:
            return False, False, [], []
        elif r_mw < p_mw:
            # 尝试配平
            # 构造反应SMILES
            to_be_balanced = True
            rxn_SMILES = ".".join(r_smiles_list) + ">>" + ".".join(p_smiles_list)
            try:
                finaldf = masterbalance(rxn_SMILES,ncpus=1)
                balanced_rxn_SMILES = finaldf["balrxnsmiles"].values[0]
                if len(balanced_rxn_SMILES.split(">>")) == 2:
                    r_smiles_list = balanced_rxn_SMILES.split(">>")[0].split(".")
                    p_smiles_list = balanced_rxn_SMILES.split(">>")[1].split(".")
                else:
                    print("\n【配平后无底物或产物】")
                    return to_be_balanced, False, [], []
            except Exception as e:
                print("【配平失败】")
                return to_be_balanced, False, [], []

        r_list = []

        for smiles in r_smiles_list:
            r_list.append({
                'role': 'substrate',
                "smiles": smiles
            })
        for smiles in p_smiles_list:
            r_list.append({
                'role': 'product',
                "smiles": smiles
            })

        reaction_list = [r_list]
        # start = time.time()
        try:
            result_am = data_process(reaction_list, localmapper)
        except:
            return to_be_balanced, False, [], []

        if result_am == 400:
            return to_be_balanced, False, [], []
        else:
            result_am = result_am[0]
            sub_list = []
            prod_list = []
            for item in result_am:
                if "role" in item.keys():
                    if "mapped_smiles" not in item.keys():
                        return to_be_balanced, False, [], []
                    else:
                        if item["role"] == "substrate":
                            sub_list.append(item["mapped_smiles"])
                        if item["role"] == "product":
                            prod_list.append(item["mapped_smiles"])
        return to_be_balanced, True, sub_list, prod_list

    def _get_bond(self):
        r_list = self.am_sub_list
        p_list = self.am_prod_list
        if len(r_list) == 0 or len(p_list) == 0:
            return False, [], [], [], [], [], None

        try:
            rxn = RXN(r_list, p_list, am=False)
        except Exception as e:
            logger.error("构造RXN对象出错：{}".format(e))
            return False, [], [], [], [], [], None

        try:
            broken, formed, bond_changed = rxn.get_bond_change(with_idx=True)
            broken_each_reactant, formed_each_product = rxn.get_bond_change_for_each()

            # 对broken, formed, bond_changed分别去重
            all_broken = list(set(broken))
            all_formed = list(set(formed))
            all_bond_changed = list(set(bond_changed))
            
            # 每个底物的断键
            broken_each_reactant_list = []
            for idx, item in enumerate(broken_each_reactant):
                if len(item) > 0:
                    item = list(set(item))
                    broken_each_reactant_list.append(item)
                else:
                    broken_each_reactant_list.append(None)

            # 每个产物的成键
            formed_each_product_list = []
            for idx, item in enumerate(formed_each_product):
                if len(item) > 0:
                    item = list(set(item))
                    formed_each_product_list.append(item)
                else:
                    formed_each_product_list.append(None)
            
            return True, all_broken, all_formed, all_bond_changed, broken_each_reactant_list, formed_each_product_list, rxn
        except Exception as e:
            logger.error("get_bond_change出错：{}".format(e))
            return False, [], [], [], [], [], None

    def _visualize_mol_with_am(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        # 创建一个绘图选项对象
        options = Draw.MolDrawOptions()
        # 创建一个绘图对象，例如使用SVG绘制
        drawer = Draw.MolDraw2DSVG(600, 300)  # 设置画布大小为600x300像素
        drawer.SetDrawOptions(options)
        # 绘制分子
        drawer.DrawMolecule(mol)
        # 完成绘图
        drawer.FinishDrawing()
        # 获取SVG数据
        svg = drawer.GetDrawingText()
        # 显示SVG或者保存到文件
        display(SVG(svg))

    def _find_bond_relationships(self, target_bond_list, target_fg_list):
        # 定义list类型的变量relationships，初始化为包含len(target_bond_list)个空list
        relationships = [[] for _ in range(len(target_bond_list))]

        for reactant_index, broken_bonds in enumerate(target_bond_list):
            fg_info = target_fg_list[reactant_index]
            if broken_bonds is not None and len(broken_bonds) > 0:
                for bond in broken_bonds:
                    if "->" in bond:
                        atoms = [atom.split(":")[1] for atom in bond.split(" -> ")[0].split(" ") if ('H' not in atom) and (":" in atom)]
                        atoms_indices = [int(atom) for atom in atoms]
                    else:
                        atoms = [atom.split(":")[1] for atom in bond.split(" ") if ('H' not in atom) and (":" in atom)]
                        atoms_indices = [int(atom) for atom in atoms]

                    # Determine the relationship of the bond to the functional groups
                    relationship = None
                    fg_names = []

                    for fg_name, atom_groups in fg_info.items():
                        for group in atom_groups:
                            # Case 1: Both atoms are in the same functional group
                            if all([atom in group for atom in atoms_indices]):
                                # relationship = "Both atoms in the same functional group"
                                relationship = 1
                                fg_names.append(fg_name)
                                # break
                    if not relationship:
                        for fg_name, atom_groups in fg_info.items():
                            for group in atom_groups:
                                # Case 1: Both atoms are in the same functional group
                                # Case 2: One atom is in a functional group, the other is not
                                if any([atom in group for atom in atoms_indices]):
                                    # relationship = "One atom in a functional group, the other not"
                                    relationship = 2
                                    fg_names.append(fg_name)
                            # if relationship:
                            #     break

                    # Case 3: Atoms are in different functional groups
                    if not relationship:
                        for i, atom1 in enumerate(atoms_indices):
                            for fg_name1, atom_groups1 in fg_info.items():
                                for group1 in atom_groups1:
                                    if atom1 in group1:
                                        for atom2 in atoms_indices[i+1:]:
                                            for fg_name2, atom_groups2 in fg_info.items():
                                                for group2 in atom_groups2:
                                                    if atom2 in group2 and fg_name1 != fg_name2:
                                                        # relationship = "Atoms in different functional groups"
                                                        relationship = 3
                                                        fg_names.extend([fg_name1, fg_name2])

                    # Case 4: Neither atom is in any functional group
                    if not relationship:
                        # relationship = "Neither atom in any functional group"
                        relationship = 4

                    # Add the relationship to the results
                    relationships[reactant_index].append({
                        'bond': bond,
                        'functional_groups': fg_names,
                        'relationship': relationship,
                    })

        return relationships

    def _extract_used_numbers(self, molecules):
        used_numbers = set()
        for mol in molecules:
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num != 0:
                    used_numbers.add(map_num)
        return used_numbers

    def _assign_new_numbers(self, molecules, used_numbers):
        max_used_number = max(used_numbers, default=0)
        for mol in molecules:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    max_used_number += 1
                    atom.SetAtomMapNum(max_used_number)
                    used_numbers.add(max_used_number)
        return molecules, used_numbers

    def show_result(self):
        display(SVG(draw_chemical_reaction(self.smiles_am,  highlightByReactant=True)))

        if self.to_be_balanced:
            print("\n原反应底物分子量小于产物分子量")

        if len(self.all_bond_changed) > 0:
            print("\n【键类型改变】")
            for bond_change in self.all_bond_changed:
                print(bond_change)

        if self.bond_status:
            for idx, item in enumerate(self.broken_each_reactant_list):
                print("\n【底物{}断键】".format(idx+1))
                self._visualize_mol_with_am(self.am_sub_list[idx])
                if item is not None and len(item) > 0:
                    for bond in item:
                        if "->" in bond:
                            continue
                        else:
                            print(bond)
            for idx, item in enumerate(self.formed_each_product_list):
                print("\n【产物{}成键】".format(idx+1))
                self._visualize_mol_with_am(self.am_prod_list[idx])
                if item is not None and len(item) > 0:
                    for bond in item:
                        print(bond)

            print("\n[断键与官能团的关系]")
            for relationship in self.broken_bond_fg_relationships:
                print(relationship)
            print("\n[成键与官能团的关系]")
            for relationship in self.formed_bond_fg_relationships:
                print(relationship)

            print("\n【官能团兼容性】")
            print("条件编号: {}".format(self.condition_id))
            print(", ".join(self.common_fg))

            if len(self.reagent_list) > 0:
                print("\n[Reagent]")
                print(", ".join(self.reagent_list))
            if len(self.solvent_list) > 0:
                print("\n[Solvent]")
                print(", ".join(self.solvent_list))
        else:
            print("\n【获取断键成键失败】")

    def process_row(self, localmapper):
        row = self.row
        self.condition_id = self.row[self.condition_id_col_name]
        # atom mapping 
        self.to_be_balanced, self.am_status, self.am_sub_list, self.am_prod_list = self._get_atom_mapping(row, localmapper)

        # Convert SMILES to RDKit Molecule objects
        am_sub_mols = [Chem.MolFromSmiles(smiles) for smiles in self.am_sub_list]
        am_prod_mols = [Chem.MolFromSmiles(smiles) for smiles in self.am_prod_list]

        # 为self.am_sub_list, self.am_prod_list中的SMILES检查是否所有原子都已有atom mapping，如果没有则在整个反应内添加atom mapping 编号
        used_numbers = self._extract_used_numbers(am_sub_mols + am_prod_mols)
        am_sub_mols, used_numbers = self._assign_new_numbers(am_sub_mols, used_numbers)
        am_prod_mols, used_numbers = self._assign_new_numbers(am_prod_mols, used_numbers)

        # Convert RDKit Molecule objects back to SMILES
        new_am_sub_list = [Chem.MolToSmiles(mol, True) for mol in am_sub_mols]
        new_am_prod_list = [Chem.MolToSmiles(mol, True) for mol in am_prod_mols]

        self.am_sub_list = new_am_sub_list
        self.am_prod_list = new_am_prod_list

        if self.am_status:
            # 获取断键成键
            self.bond_status, self.all_broken, self.all_formed, self.all_bond_changed, self.broken_each_reactant_list, self.formed_each_product_list, rxn_obj = self._get_bond()
            self.smiles_am = rxn_obj.rxn_smiles
        else:
            self.bond_status = False
            print("\n【获取atom mapping失败，暂时无法识别断键成键】")
            

        if self.bond_status:
            # 官能团信息
            # 底物官能团信息
            fg_in_sub = []
            fg_in_prod = []
            r_large_fg_list = []
            p_large_fg_list = []
            for idx, smiles in enumerate(self.am_sub_list):
                large_fg = Large_FG(smiles, smiles_with_atom_mapping=True)
                result = large_fg.find_FG(level=2)["data"]
                r_large_fg_list.append(large_fg)
                self.fg_each_reactant_list.append(result)
                for key, value in result.items():
                    fg_in_sub.append(key)
            fg_in_sub = list(set(fg_in_sub))
            for idx, smiles in enumerate(self.am_prod_list):
                large_fg = Large_FG(smiles, smiles_with_atom_mapping=True)
                result = large_fg.find_FG(level=2)["data"]
                p_large_fg_list.append(large_fg)
                self.fg_each_product_list.append(result)
                for key, value in result.items():
                    fg_in_prod.append(key)
            fg_in_prod = list(set(fg_in_prod))

            try:
                self.broken_bond_fg_relationships = self._find_bond_relationships(self.broken_each_reactant_list, self.fg_each_reactant_list)
                self.formed_bond_fg_relationships = self._find_bond_relationships(self.formed_each_product_list, self.fg_each_product_list)
            except Exception as e:
                print(self.broken_each_reactant_list)
                print(self.fg_each_reactant_list)
                print(self.formed_each_product_list)
                print(self.fg_each_product_list)

            # 官能团兼容性
            # 找出fg_in_prod和fg_in_sub中都包含的官能团
            self.common_fg = list(set(fg_in_sub) & set(fg_in_prod))
            # 反应条件
            self.reagent_list = []
            for reagent in self.reagent_col_name_list:
                if row[reagent] is not None and pd.notna(row[reagent]) and row[reagent] != "":
                    self.reagent_list.append(row[reagent])
            self.solvent_list = []
            for solvent in self.solvent_col_name_list:
                if row[solvent] is not None and pd.notna(row[solvent]) and row[reagent] != "":
                    self.solvent_list.append(row[solvent])

class Factory:
    def __init__(
            self,
            df,
            condition_df,
            sub_col_name_list = ["底物1", "底物2", "底物3", "底物4", "底物5", "底物6"],
            prod_col_name_list = ["产物1", "产物2", "产物3", "产物4", "产物5", "产物6", "产物7"],
            reagent_col_name_list = ["Reagent1", "Reagent2", "Reagent3", "Reagent4", "Reagent5", "Reagent6", "Reagent7",
                                     "Reagent8", "Reagent9", "Reagent10", "Reagent11", "Reagent12", "Reagent13", "Reagent14",
                                     "Reagent15", "Reagent16", "Reagent17", "Reagent18", "Reagent19", "Reagent20", "Reagent21"],
            solvent_col_name_list = ["Solvent1", "Solvent2", "Solvent3", "Solvent4", "Solvent5", "Solvent6", "Solvent7",
                                     "Solvent8", "Solvent9", "Solvent10"],
            cond_id_col_name = "条件编号",
            other_cond_col_name_list = [],
            freq_col_name = "数据频次",
            data_path = ""
        ):
        self.data_path = data_path
        # 初始化列名
        self.sub_col_name_list = sub_col_name_list
        self.prod_col_name_list = prod_col_name_list
        self.reagent_col_name_list = reagent_col_name_list
        self.solvent_col_name_list = solvent_col_name_list
        self.other_cond_col_name_list = other_cond_col_name_list

        self.localmapper = localmapper()
        self.df = df

        # 去重
        print("去重前：{}".format(self.df.shape))
        self.df = self.df.drop_duplicates(subset=sub_col_name_list+prod_col_name_list+reagent_col_name_list+solvent_col_name_list+other_cond_col_name_list)
        print("去重后：{}".format(self.df.shape))
        
        self.df = self.df.reset_index(drop=True)
        self.df["idx"] = self.df.index + 1

        # 如果输入的condition_df为None，则自动生成condition_df，并设置cond_id_col_name和
        if condition_df is not None:
            self.condition_df = condition_df
            self.condition_id_col_name = cond_id_col_name
            self.freq_col_name = freq_col_name
        else:
            self.data_table_path, self.condition_table_path, self.condition_df = self._generate_condition_table()
            self.condition_id_col_name = "条件编号"
            self.freq_col_name = "数据频次"

        self.condition_dict, self.all_reagent_list, self.all_solvent_list = self.pre_process_condition()
        self.data = [None] * len(self.df)
        self.stat_df = pd.DataFrame()
        self.broken_col_names = []
        self.formed_col_names = []
        self.changed_col_names = []
        self.r_fg_cols = []
        self.p_fg_cols = []
        self.broken_to_formed_dict = {}
        self.formed_to_broken_dict = {}
        self.all_formed_dict = {}
        self.all_broken_dict = {}
        self.hierarchical_formed_dict = {}
        self.hierarchical_broken_dict = {}

    def _generate_condition_table(self):
        target_cols = self.reagent_col_name_list + self.solvent_col_name_list + self.other_cond_col_name_list
        for col_name in target_cols:
            # 将df[col_name]中的每个元素转换为str类型
            self.df[col_name] = self.df[col_name].astype(str)
            # 将df[col_name]的所有"NAN"替换为""
            self.df[col_name] = self.df[col_name].replace("nan", "")
            self.df[col_name] = self.df[col_name].replace("NAN", "")
        condition_df = self.df[target_cols]
        condition_id = 1
        for name, group in condition_df.groupby(target_cols):
            condition_df.loc[group.index, '条件编号'] = condition_id
            condition_df.loc[group.index, '数据频次'] = len(group)
            self.df.loc[group.index, '条件编号'] = condition_id
            condition_id += 1
        df_drop_dup = condition_df.drop_duplicates(subset=target_cols)
        # 保存到xlsx文件
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        data_path = "{}/data-with_cond_id-{}.xlsx".format(self.data_path, timestamp)
        cond_path = "{}/condition_df-drop_dup-{}.xlsx".format(self.data_path, timestamp)
        #condition_df是df子集
        self.df.to_excel(data_path, index=False)
        condition_df.to_excel(cond_path, index=False)
        return data_path, cond_path, condition_df

    def pre_process_condition(self):
        condition_dict = {}
        all_reagent_list = []
        all_solvent_list = []
        for index, row in self.condition_df.iterrows():
            condition_id = row[self.condition_id_col_name]
            frequency = row[self.freq_col_name]
            other_cond_list = []
            if len(self.other_cond_col_name_list) > 0:
                for other_cond in self.other_cond_col_name_list:
                    if row[other_cond] is not None and pd.notna(row[other_cond]):
                        other_cond_list.append(row[other_cond])
            reagent_list = []
            solvent_list = []
            for reagent in self.reagent_col_name_list:
                if row[reagent] is not None and pd.notna(row[reagent]) and row[reagent] != "":
                    reagent_list.append(str(row[reagent]))
            for solvent in self.solvent_col_name_list:
                if row[solvent] is not None and pd.notna(row[solvent]) and row[reagent] != "":
                    solvent_list.append(str(row[solvent]))
            result = {
                "frequency": frequency,
                "reagent": reagent_list,
                "solvent": solvent_list,
                "other_cond": other_cond_list
            }
            all_reagent_list += reagent_list
            all_reagent_list = list(set(all_reagent_list))
            all_solvent_list += solvent_list
            all_solvent_list = list(set(all_solvent_list))
            condition_dict[condition_id] = result
        return condition_dict, all_reagent_list, all_solvent_list

    def get_stat_info(self, filter_name:str, value):
        if filter_name == "condition_id":
            condition_id = value
            logger.info("条件编号: {}".format(condition_id))
            # 从self.stat_df筛选出条件对应的行
            if condition_id != 0:
                target_df = self.stat_df[(self.stat_df["condition_idx"] == condition_id)]
                # target_df = self.stat_df[(self.stat_df["condition_idx"] == condition_id) & (self.stat_df["r_mw<p_mw"] == False)]
            else:
                target_df = self.stat_df
                # target_df = self.stat_df[(self.stat_df["r_mw<p_mw"] == False)]
        elif filter_name == "reagent":
            reagent_name = value
            logger.info("Reagent: {}".format(reagent_name))
            # 从self.stat_df筛选出条件对应的行
            contains_reagent = self.stat_df["reagent_str"].str.contains(reagent_name.replace("(", "\(").replace(")", "\)"), na=False)
            target_df = self.stat_df[contains_reagent]
            # target_df = target_df[target_df["r_mw<p_mw"] == False]
            print("使用{}的反应总数：{}".format(reagent_name, len(target_df)))
        elif filter_name == "solvent":
            solvent_name = value
            logger.info("Solvent: {}".format(solvent_name))
            # 从self.stat_df筛选出条件对应的行
            contains_solvent = self.stat_df["solvent_str"].str.contains(solvent_name.replace("(", "\(").replace(")", "\)"), na=False)
            target_df = self.stat_df[contains_solvent]
            # target_df = target_df[target_df["r_mw<p_mw"] == False]
            print("使用{}的反应总数：{}".format(solvent_name, len(target_df)))

        # 断键统计
        broken_sum = target_df[self.broken_col_names].sum()
        broken_sum = broken_sum[broken_sum!=0].sort_values(ascending=False)
        if filter_name == "condition_id" and value != 0:
            condition_id = value
            print("条件编号{}的断键成键统计:".format(condition_id))
            if len(self.condition_dict[condition_id]["reagent"]) > 0:
                print("[Reagent]")
                print(", ".join(self.condition_dict[condition_id]["reagent"]))
            if len(self.condition_dict[condition_id]["solvent"]) > 0:
                print("[Solvent]")
                print(", ".join(self.condition_dict[condition_id]["solvent"]))
            print("条件{}出现频次:{}".format(condition_id, self.condition_dict[condition_id]["frequency"]))

        if len(broken_sum) > 0:
            print("\n[断键统计]")
            for idx, row in broken_sum.items():
                print("{}:\t{}".format(idx.split(":")[1], row))

        formed_sum = target_df[self.formed_col_names].sum()
        formed_sum = formed_sum[formed_sum!=0].sort_values(ascending=False)
        if len(formed_sum) > 0:
            print("\n[成键统计]")
            for idx, row in formed_sum.items():
                print("{}:\t{}".format(idx.split(":")[1], row))
        
        changed_sum = target_df[self.changed_col_names].sum()
        changed_sum = changed_sum[changed_sum!=0].sort_values(ascending=False)
        if len(changed_sum) > 0:
            print("\n[键类型改变统计]")
            for idx, row in changed_sum.items():
                print("{}:\t{}".format(idx.split(":")[1], row))

        # 官能团兼容性
        try:
            sets = target_df['fg_compatibility'].apply(lambda x: {} if (x is None) or (pd.isna(x)) else set(x.split(',')))
            if len(list(sets)) == 0:
                fg_compatibility = []
            else:
                # 使用set.intersection方法找出所有集合的交集
                fg_compatibility = list(set.intersection(*sets))
        except Exception as e:
            logger.error(e)
            logger.error(sets)
            fg_compatibility = []
        if len(fg_compatibility) > 0:
            print("\n[官能团兼容性]")
            print(", ".join(fg_compatibility))

    def select_row_for_condition(self):
        def handle_select_multiple_change(change):
            # 清楚原来的输出
            clear_output()
            # 重新显示下拉框
            display(self.drop_down_for_condition)
            condition_id = change.new
            self.get_stat_info(filter_name = "condition_id", value = condition_id)

        self.drop_down_for_condition = widgets.Dropdown(
            options=[0]+list(self.condition_dict.keys()),
            description='条件编号:',
            disabled=False,
        )
        self.drop_down_for_condition.observe(handle_select_multiple_change, names='value')
        display(self.drop_down_for_condition)

    def select_row_for_reagent(self):
        def handle_select_multiple_change(change):
            # 清楚原来的输出
            clear_output()
            # 重新显示下拉框
            display(self.drop_down_for_reagent)
            reagent_name = change.new
            self.get_stat_info(filter_name = "reagent", value = reagent_name)

        self.drop_down_for_reagent = widgets.Dropdown(
            options=self.all_reagent_list,
            description='Reagent:',
            disabled=False,
        )
        self.drop_down_for_reagent.observe(handle_select_multiple_change, names='value')
        display(self.drop_down_for_reagent)

    def select_row_for_solvent(self):
        def handle_select_multiple_change(change):
            # 清楚原来的输出
            clear_output()
            # 重新显示下拉框
            display(self.drop_down_for_solvent)
            solvent_name = change.new
            self.get_stat_info(filter_name = "solvent", value = solvent_name)

        self.drop_down_for_solvent = widgets.Dropdown(
            options=self.all_solvent_list,
            description='Solvent:',
            disabled=False,
        )
        self.drop_down_for_solvent.observe(handle_select_multiple_change, names='value')
        display(self.drop_down_for_solvent)

    def get_comb_by_broken_fromed_bond(self, bond, broken_or_formed, sub_idx_list = []):
        result_list = []
        if broken_or_formed == "broken":
            print("目标断键：{}".format(bond))
            # print("相关反应：\n{}".format(", ".join(self.all_broken_dict[bond]["rxn_row_number"])))
            print("相关反应：\n")
            counter = 0
            for row_idx in self.all_broken_dict[bond]["rxn_row_number"]:
                if len(sub_idx_list) > 0:
                    if int(row_idx)-1 not in sub_idx_list:
                        continue
                if counter > 5:
                    break
                this_row_idx = int(row_idx) - 1
                print("反应编号: {}".format(int(row_idx)))
                if self.data[this_row_idx] is not None:
                    if self.data[this_row_idx].am_status == False:
                        print("该反应atom mapping 失败")
                    elif self.data[this_row_idx].bond_status == False:
                        print("未能识别该反应的断键成键及官能团")
                    else:
                        self.data[this_row_idx].show_result()
                        counter += 1
            print("相关Reagent：\n{}".format(self.all_broken_dict[bond]["reagent_combination_list"]))
            print("相关Solvent：\n{}".format(self.all_broken_dict[bond]["solvent_combination_list"]))                  
            # print("已有成键组合：")
            # target_list = self.broken_to_formed_dict
        elif broken_or_formed == "formed":
            print("目标成键：{}".format(bond))
            print("相关反应：\n{}".format(", ".join(self.all_formed_dict[bond]["rxn_row_number"])))
            print("相关Reagent：\n{}".format(self.all_formed_dict[bond]["reagent_combination_list"]))
            print("相关Solvent：\n{}".format(self.all_formed_dict[bond]["solvent_combination_list"]))                  
            # print("已有断键组合：")
            # target_list = self.formed_to_broken_dict
        else:
            return []
        # for key in target_list.keys():
        #     if bond in key:
        #         result_list += target_list[key]["combination_list"]
        # if len(result_list) > 0:
        #     for item in result_list:
        #         print(item)

    def select_row_for_broken_bond(self):
        def handle_select_multiple_change(change):
            # 清楚原来的输出
            clear_output()
            # 重新显示下拉框
            display(self.drop_down_for_broken_bond)
            bond = change.new
            self.get_comb_by_broken_fromed_bond(bond = bond, broken_or_formed = "broken")

        self.drop_down_for_broken_bond = widgets.Dropdown(
            options=sorted(self.all_broken_dict.keys()),
            description='断键:',
            disabled=False,
        )
        self.drop_down_for_broken_bond.observe(handle_select_multiple_change, names='value')
        display(self.drop_down_for_broken_bond)

    def select_row_for_formed_bond(self):
        def handle_select_multiple_change(change):
            # 清楚原来的输出
            clear_output()
            # 重新显示下拉框
            display(self.drop_down_for_formed_bond)
            bond = change.new
            self.get_comb_by_broken_fromed_bond(bond = bond, broken_or_formed = "formed")

        self.drop_down_for_formed_bond = widgets.Dropdown(
            options=sorted(self.all_formed_dict.keys()),
            description='成键:',
            disabled=False,
        )
        self.drop_down_for_formed_bond.observe(handle_select_multiple_change, names='value')
        display(self.drop_down_for_formed_bond)

    def select_row_for_rxn(self):
        def handle_select_multiple_change(change):
            # 清楚原来的输出
            clear_output()
            # 重新显示下拉框
            display(self.drop_down_for_rxn)
            row_idx = change.new - 1
            logger.info("反应编号: {}".format(row_idx))
            if self.data[row_idx] is not None:
                if self.data[row_idx].am_status == False:
                    print("该反应atom mapping 失败")
                elif self.data[row_idx].bond_status == False:
                    print("未能识别该反应的断键成键及官能团")
                else:
                    self.data[row_idx].show_result()
            else:
                row = self.df.iloc[row_idx]
                data_obj = Data_Obj(
                    row_idx,
                    row,
                    self.sub_col_name_list,
                    self.prod_col_name_list,
                    self.reagent_col_name_list,
                    self.solvent_col_name_list,
                    self.condition_id_col_name
                )
                # 开始处理相应的行
                data_obj.process_row(localmapper=self.localmapper)
                self.data[row_idx] = data_obj
                if data_obj.am_status == False:
                    print("该反应atom mapping 失败")
                elif data_obj.bond_status == False:
                    print("未能识别该反应的断键成键及官能团")
                else:
                    data_obj.show_result()
        self.drop_down_for_rxn = widgets.Dropdown(
            options=list(self.df.index+1),
            description='反应编号:',
            disabled=False,
        )
        self.drop_down_for_rxn.observe(handle_select_multiple_change, names='value')
        display(self.drop_down_for_rxn)

    def _get_bond_without_atom_idx(self, bond, changed=False):
        bond_split = bond.split(" ")
        atom_1 = bond_split[0]
        if ":" in atom_1:
            atom_1_ele = atom_1.split(":")[0]
        else:
            atom_1_ele = atom_1
        bond_type = bond_split[1]
        atom_2 = bond_split[2]
        if ":" in atom_2:
            atom_2_ele = atom_2.split(":")[0]
        else:
            atom_2_ele = atom_2
        
        if changed:
            bond_type_new = bond_split[5]
            return atom_1_ele, bond_type, atom_2_ele, bond_type_new
        else:
            return atom_1_ele, bond_type, atom_2_ele, None

    def stat_info(self, data_obj:Data_Obj, rxn_idx:int):
        result_dict = {
            "rxn_idx": rxn_idx+1,
            "condition_idx": data_obj.condition_id,
            "r_mw<p_mw": data_obj.to_be_balanced,
            "am_status": True,
            "bond_status": True,
            "rxn_SMILES": data_obj.smiles_am
        }
        broken_cols = []
        formed_cols = []
        changed_cols = []
        r_fg_cols = []
        p_fg_cols = []
        # 解析断键
        for bond in data_obj.all_broken:
            atom_1_ele, bond_type, atom_2_ele, _ = self._get_bond_without_atom_idx(bond)
            col_name = "broken:{} {} {}".format(atom_1_ele, bond_type, atom_2_ele)
            if col_name not in result_dict.keys():
                result_dict[col_name] = 1
                broken_cols.append(col_name)

        # 解析成键
        for bond in data_obj.all_formed:
            atom_1_ele, bond_type, atom_2_ele, _ = self._get_bond_without_atom_idx(bond)
            col_name = "formed:{} {} {}".format(atom_1_ele, bond_type, atom_2_ele)
            if col_name not in result_dict.keys():
                result_dict[col_name] = 1
                formed_cols.append(col_name)
            
        # 解析键类型改变
        for bond in data_obj.all_bond_changed:
            atom_1_ele, bond_type, atom_2_ele, bond_type_new = self._get_bond_without_atom_idx(bond, changed = True)
            col_name = "changed:{} {} {} -> {} {} {}".format(atom_1_ele, bond_type, atom_2_ele, atom_1_ele, bond_type_new, atom_2_ele)
            if col_name not in result_dict.keys():
                result_dict[col_name] = 1
                changed_cols.append(col_name)

        # 每个底物中包含的官能团
        for r_idx, fg_list in enumerate(data_obj.fg_each_reactant_list):
            for fg_name in fg_list:
                col_name = "r{}_{}".format(r_idx+1, fg_name)
                result_dict[col_name] = 1
                r_fg_cols.append(col_name)

        # 每个产物中包含的官能团
        for p_idx, fg_list in enumerate(data_obj.fg_each_product_list):
            for fg_name in fg_list:
                col_name = "p{}_{}".format(p_idx+1, fg_name)
                # col_name = "p{}_{}".format(p_idx, fg_name)
                result_dict[col_name] = 1
                p_fg_cols.append(col_name)

        # 所有底物中包含的官能团
        fg_in_sub = list(set(element for sublist in data_obj.fg_each_reactant_list for element in sublist))
        result_dict["fg_in_sub"] = ", ".join(fg_in_sub)

        # 所有产物中包含的官能团
        fg_in_prod = list(set(element for sublist in data_obj.fg_each_product_list for element in sublist))
        result_dict["fg_in_prod"] = ", ".join(fg_in_prod)

        # 官能团兼容性
        common_fg_str = ",".join(data_obj.common_fg)
        result_dict["fg_compatibility"] = common_fg_str

        # 将断键成键与官能团的关系组织为易于统计的结构
        def process_broken_bond_fg_relationships(data_obj, broken_or_formed):
            if broken_or_formed == "broken":
                target = data_obj.broken_bond_fg_relationships
            elif broken_or_formed == "formed":
                target = data_obj.formed_bond_fg_relationships
            result = []
            for sub_or_prod_idx,  sub_or_prod in enumerate(target):
                for relationship in sub_or_prod:
                    this_bond = relationship["bond"]
                    if "->" in this_bond:
                        continue
                    else:
                        atom_1 = this_bond.split(" ")[0]
                        if ":" in atom_1:
                            atom_1 = this_bond.split(" ")[0].split(":")[0]
                        atom_2 = this_bond.split(" ")[2]
                        if ":" in atom_2:
                            atom_2 = this_bond.split(" ")[2].split(":")[0]
                        this_bond_type = this_bond.split(" ")[1]
                        this_bond = atom_1 + " " + this_bond_type + " " + atom_2
                    this_fg = ";".join(sorted(relationship["functional_groups"]))
                    this_relationship = relationship["relationship"]
                    result.append((this_bond, this_fg, this_relationship))
            return tuple(sorted(list(set(result))))
        broken_bond_fg_relationships = process_broken_bond_fg_relationships(data_obj, "broken")
        formed_bond_fg_relationships = process_broken_bond_fg_relationships(data_obj, "formed")

        return result_dict, broken_cols, formed_cols, changed_cols, r_fg_cols, p_fg_cols, broken_bond_fg_relationships, formed_bond_fg_relationships
        # return result_dict, broken_cols, formed_cols, changed_cols, r_fg_cols, p_fg_cols
    
    def generate_all(self, callback = None):
        result_list = []    # type: list[pd.DataFrame] # 其中的元素为每个反应的处理结果
        if len(self.stat_df) > 0:
            self.stat_df = pd.DataFrame()
        # result_list.append(self.stat_df)
        for index, row in tqdm(self.df.iterrows()):
            if self.data[index] is None:
                data_obj = Data_Obj(
                    index,
                    row,
                    self.sub_col_name_list,
                    self.prod_col_name_list,
                    self.reagent_col_name_list,
                    self.solvent_col_name_list,
                    self.condition_id_col_name
                )
                # 开始处理一条反应数据的行
                data_obj.process_row(localmapper=self.localmapper)
                self.data[index] = data_obj

            if self.data[index].am_status == False:
                new_row = pd.Series({
                    "rxn_idx": index+1,
                    "condition_idx": self.data[index].condition_id,
                    "r_mw<p_mw": self.data[index].to_be_balanced,
                    "am_status": False,
                    "bond_status": False
                })
            elif self.data[index].bond_status == False:
                new_row = pd.Series({
                    "rxn_idx": index+1,
                    "condition_idx": self.data[index].condition_id,
                    "r_mw<p_mw": self.data[index].to_be_balanced,
                    "am_status": True,
                    "bond_status": False
                })
            else:
                stat_data, broken_cols, formed_cols, changed_cols, r_fg_cols, p_fg_cols, broken_bond_fg_relationships, formed_bond_fg_relationships = self.stat_info(self.data[index], rxn_idx=index)
                # stat_data, broken_cols, formed_cols, changed_cols, r_fg_cols, p_fg_cols = self.stat_info(data_obj, rxn_idx=index)

                new_row = pd.Series(stat_data)
                self.broken_col_names += broken_cols
                self.formed_col_names += formed_cols
                self.changed_col_names += changed_cols
                self.r_fg_cols += r_fg_cols
                self.p_fg_cols += p_fg_cols
                self.broken_col_names = list(set(self.broken_col_names))
                self.formed_col_names = list(set(self.formed_col_names))
                self.changed_col_names = list(set(self.changed_col_names))
                self.r_fg_cols = list(set(self.r_fg_cols))
                self.p_fg_cols = list(set(self.p_fg_cols))

                condition_id = self.data[index].condition_id
                reagent_list = self.condition_dict[condition_id]["reagent"]
                solvent_list = self.condition_dict[condition_id]["solvent"]

                # 断键成键组合统计
                # 成键 -> 断键
                if formed_bond_fg_relationships in self.formed_to_broken_dict.keys():
                    self.formed_to_broken_dict[formed_bond_fg_relationships]["combination_list"].append(broken_bond_fg_relationships)
                    self.formed_to_broken_dict[formed_bond_fg_relationships]["rxn_row_number"].append(str(index+1))
                    self.formed_to_broken_dict[formed_bond_fg_relationships]["reagent_combination_list"].append(reagent_list)
                    self.formed_to_broken_dict[formed_bond_fg_relationships]["solvent_combination_list"].append(solvent_list)
                else:
                    self.formed_to_broken_dict[formed_bond_fg_relationships] = {
                        "combination_list": [broken_bond_fg_relationships],
                        "rxn_row_number": [str(index+1)],
                        "reagent_combination_list": [reagent_list],
                        "solvent_combination_list": [solvent_list]
                    }
                # 去重
                def drop_dup_list(list_of_lists):
                    # 使用集合去重，首先将内部列表转换为元组
                    unique_tuples = set(tuple(x) for x in list_of_lists)
                    # 现在将元组转换回列表
                    unique_list_of_lists = [list(x) for x in unique_tuples]
                    return unique_list_of_lists

                self.formed_to_broken_dict[formed_bond_fg_relationships]["combination_list"] = list(set(self.formed_to_broken_dict[formed_bond_fg_relationships]["combination_list"]))
                self.formed_to_broken_dict[formed_bond_fg_relationships]["reagent_combination_list"] = drop_dup_list(self.formed_to_broken_dict[formed_bond_fg_relationships]["reagent_combination_list"])
                self.formed_to_broken_dict[formed_bond_fg_relationships]["solvent_combination_list"] = drop_dup_list(self.formed_to_broken_dict[formed_bond_fg_relationships]["solvent_combination_list"])

                # 断键 -> 成键
                if broken_bond_fg_relationships in self.broken_to_formed_dict.keys():
                    self.broken_to_formed_dict[broken_bond_fg_relationships]["combination_list"].append(formed_bond_fg_relationships)
                    self.broken_to_formed_dict[broken_bond_fg_relationships]["rxn_row_number"].append(str(index+1))
                    self.broken_to_formed_dict[broken_bond_fg_relationships]["reagent_combination_list"].append(reagent_list)
                    self.broken_to_formed_dict[broken_bond_fg_relationships]["solvent_combination_list"].append(solvent_list)
                else:
                    self.broken_to_formed_dict[broken_bond_fg_relationships] = {
                        "combination_list": [formed_bond_fg_relationships],
                        "rxn_row_number": [str(index+1)],
                        "reagent_combination_list": [reagent_list],
                        "solvent_combination_list": [solvent_list]
                    }
                # 去重
                self.broken_to_formed_dict[broken_bond_fg_relationships]["combination_list"] = list(set(self.broken_to_formed_dict[broken_bond_fg_relationships]["combination_list"]))
                self.broken_to_formed_dict[broken_bond_fg_relationships]["reagent_combination_list"] = drop_dup_list(self.broken_to_formed_dict[broken_bond_fg_relationships]["reagent_combination_list"])
                self.broken_to_formed_dict[broken_bond_fg_relationships]["solvent_combination_list"] = drop_dup_list(self.broken_to_formed_dict[broken_bond_fg_relationships]["solvent_combination_list"])

            new_row = pd.DataFrame(new_row).T
            result_list.append(new_row)
            # callback(((((index+1) / len(self.df))*79) + 10)/100)
        print("generate_all done")
        self.stat_df = pd.concat(result_list, ignore_index=True)
        print("stat_df concat done")

    def get_all_broken_formed_bond(self):
        all_broken_tuple_list = list(self.broken_to_formed_dict.keys())
        all_formed_tuple_list = list(self.formed_to_broken_dict.keys())
        # 遍历all_broken_tuple_list中的所有tuple，找出所有的断键
        all_broken_dict = {}
        for item in all_broken_tuple_list:
            for bond in item:
                if bond in all_broken_dict.keys():
                    all_broken_dict[bond]["rxn_row_number"] += self.broken_to_formed_dict[item]["rxn_row_number"]
                    all_broken_dict[bond]["reagent_combination_list"] += self.broken_to_formed_dict[item]["reagent_combination_list"]
                    all_broken_dict[bond]["solvent_combination_list"] += self.broken_to_formed_dict[item]["solvent_combination_list"]
                else:
                    all_broken_dict[bond] = {
                        "rxn_row_number": self.broken_to_formed_dict[item]["rxn_row_number"],
                        "reagent_combination_list": self.broken_to_formed_dict[item]["reagent_combination_list"],
                        "solvent_combination_list": self.broken_to_formed_dict[item]["solvent_combination_list"]
                    }
        def drop_dup_list(list_of_lists):
            # 使用集合去重，首先将内部列表转换为元组
            unique_tuples = set(tuple(x) for x in list_of_lists)
            # 现在将元组转换回列表
            unique_list_of_lists = [list(x) for x in unique_tuples]
            return unique_list_of_lists
        # 对all_broken_dict中每个键值对的值进行去重
        for key in all_broken_dict.keys():
            all_broken_dict[key]["rxn_row_number"] = list(set(all_broken_dict[key]["rxn_row_number"]))
            all_broken_dict[key]["reagent_combination_list"] = drop_dup_list(all_broken_dict[key]["reagent_combination_list"])
            all_broken_dict[key]["solvent_combination_list"] = drop_dup_list(all_broken_dict[key]["solvent_combination_list"])
        self.all_broken_dict = all_broken_dict
        # 遍历all_formed_tuple_list中的所有tuple，找出所有的成键
        all_formed_dict = {}
        for item in all_formed_tuple_list:
            for bond in item:
                if bond in all_formed_dict.keys():
                    all_formed_dict[bond]["rxn_row_number"] += self.formed_to_broken_dict[item]["rxn_row_number"]
                    all_formed_dict[bond]["reagent_combination_list"] += self.formed_to_broken_dict[item]["reagent_combination_list"]
                    all_formed_dict[bond]["solvent_combination_list"] += self.formed_to_broken_dict[item]["solvent_combination_list"]
                else:
                    all_formed_dict[bond] = {
                        "rxn_row_number": self.formed_to_broken_dict[item]["rxn_row_number"],
                        "reagent_combination_list": self.formed_to_broken_dict[item]["reagent_combination_list"],
                        "solvent_combination_list": self.formed_to_broken_dict[item]["solvent_combination_list"]
                    }
        # 对all_formed_dict中每个键值对的值进行去重
        for key in all_formed_dict.keys():
            all_formed_dict[key]["rxn_row_number"] = list(set(all_formed_dict[key]["rxn_row_number"]))
            all_formed_dict[key]["reagent_combination_list"] = drop_dup_list(all_formed_dict[key]["reagent_combination_list"])
            all_formed_dict[key]["solvent_combination_list"] = drop_dup_list(all_formed_dict[key]["solvent_combination_list"])
        self.all_formed_dict = all_formed_dict
    
    def get_outline_broken_formed_bond(self):
        for key in self.all_broken_dict.keys():
            bond = key[0]
            fg = key[1]
            relation = key[2]
            if bond in self.hierarchical_broken_dict.keys():
                if fg in self.hierarchical_broken_dict[bond].keys():
                    if relation not in self.hierarchical_broken_dict[bond][fg]:
                        self.hierarchical_broken_dict[bond][fg].append(relation)
                else:
                    self.hierarchical_broken_dict[bond][fg] = [relation]
            else:
                self.hierarchical_broken_dict[bond] = {fg: [relation]}
        for key in self.all_formed_dict.keys():
            bond = key[0]
            fg = key[1]
            relation = key[2]
            if bond in self.hierarchical_formed_dict.keys():
                if fg in self.hierarchical_formed_dict[bond].keys():
                    if relation not in self.hierarchical_formed_dict[bond][fg]:
                        self.hierarchical_formed_dict[bond][fg].append(relation)
                else:
                    self.hierarchical_formed_dict[bond][fg] = [relation]
            else:
                self.hierarchical_formed_dict[bond] = {fg: [relation]}

    def save_all_reagent_solvent_in_rxn_to_col(self):
        for idx, row in self.stat_df.iterrows():
            condition_id = row["condition_idx"]
            try:
                reagent_str = ", ".join(self.condition_dict[condition_id]["reagent"])
            except:
                print(self.condition_dict[condition_id]["reagent"])
            solvent_str = ", ".join(self.condition_dict[condition_id]["solvent"])
            self.stat_df.loc[idx, "reagent_str"] = reagent_str
            self.stat_df.loc[idx, "solvent_str"] = solvent_str
    
    def save_data(self):
        print("开始保存完整数据")
        data_to_dump = {
            "data": self.data,
            "stat_df": self.stat_df,
            "broken_col_names": self.broken_col_names,
            "formed_col_names": self.formed_col_names,
            "changed_col_names": self.changed_col_names,
            "r_fg_cols": self.r_fg_cols,
            "p_fg_cols": self.p_fg_cols,
            "broken_to_formed_dict": self.broken_to_formed_dict,
            "formed_to_broken_dict": self.formed_to_broken_dict,
            "all_broken_dict": self.all_broken_dict,
            "all_formed_dict": self.all_formed_dict,
            "hierarchical_broken_dict": self.hierarchical_broken_dict,
            "hierarchical_formed_dict": self.hierarchical_formed_dict
        }

        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        file_path = "{}/data-{}.pkl".format(self.data_path, now)
        with open(file_path, "wb") as f:
            pickle.dump(data_to_dump, f)
        print("数据保存完毕")
        return file_path

    def save_stat_df_to_csv(self):
        print("开始保存stat_df数据")
        # 重新排序列名
        new_col_name = ["rxn_idx", "condition_idx", "r_mw<p_mw", "am_status",
                         "bond_status", "rxn_SMILES", "fg_compatibility", "fg_in_prod", "fg_in_sub"]

        if "reagent_str" in self.stat_df.columns:
            new_col_name.append("reagent_str")
        if "solvent_str" in self.stat_df.columns:
            new_col_name.append("solvent_str")

        col_name_need_to_sort = list(self.stat_df.columns)
        for col_name in new_col_name:
            try:
                col_name_need_to_sort.remove(col_name)
            except Exception as e:
                print(e)
                print(col_name)

        col_name_need_to_sort = sorted(col_name_need_to_sort)
        new_col_name += col_name_need_to_sort
        stat_df_to_save = self.stat_df[new_col_name]

        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        file_path = "{}/stat_data-{}.csv".format(self.data_path, now)
        stat_df_to_save.to_csv(file_path, index=False, encoding="utf_8_sig")
        print("stat_df数据保存完毕")
        return file_path

    def load_data_from_pickle(self, file_path:str):
        with open(file_path, "rb") as file:
            loaded_data = pickle.load(file)
        self.data = loaded_data["data"]
        self.stat_df = loaded_data["stat_df"]
        self.broken_col_names = loaded_data["broken_col_names"]
        self.formed_col_names = loaded_data["formed_col_names"]
        self.changed_col_names = loaded_data["changed_col_names"]
        self.r_fg_cols = loaded_data["r_fg_cols"]
        self.p_fg_cols = loaded_data["p_fg_cols"]
        self.broken_to_formed_dict = loaded_data["broken_to_formed_dict"]
        self.formed_to_broken_dict = loaded_data["formed_to_broken_dict"]
        self.all_broken_dict = loaded_data["all_broken_dict"]
        self.all_formed_dict = loaded_data["all_formed_dict"]
        self.hierarchical_broken_dict = loaded_data["hierarchical_broken_dict"]
        self.hierarchical_formed_dict = loaded_data["hierarchical_formed_dict"]

class Fg_check:
    def __init__(self):
        self.timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.rd_func_group = FuncGroups()
        self.df = pd.read_csv("data/Ylide/SMILES_for_FG_check.csv")
        self.this_index = 0
        print(self.df.shape)
        if "check_passed" not in self.df.columns:
            self.df["check_passed"] = "not_yet"
        # 定义按钮
        self.next_batch_button = widgets.Button(description="下一批分子")
        self.next_batch_button.on_click(self.on_next_batch_clicked)
        self.true_button = widgets.Button(description = "识别正确")
        self.false_button = widgets.Button(description = "识别错误")
        self.output = widgets.Output()
        self.true_button.on_click(self.on_true_button_clicked)
        self.false_button.on_click(self.on_false_button_clicked)
        display(self.next_batch_button, self.true_button, self.false_button, self.output)

    def on_true_button_clicked(self, b):
        with self.output:
            # 将self.df index为self.index的check_passed列的值改为true
            self.df.loc[self.this_index, "check_passed"] = "True"
            self.save_csv()
            print("识别正确")

    def on_false_button_clicked(self, b):
        with self.output:
            # 将self.df index为self.index的check_passed列的值改为true
            self.df.loc[self.this_index, "check_passed"] = "False"
            self.save_csv()
            print("识别错误")

    def on_next_batch_clicked(self, b):
        with self.output:
            self.output.clear_output()
            # 从self.df["check_passed"]为None中随机取1行
            sample_df = self.df[self.df["check_passed"]=="not_yet"].sample(1)
            smiles = sample_df.iloc[0, 1]
            self.this_index = sample_df.index[0]
            print(smiles)
            large_fg = Large_FG(smiles, smiles_with_atom_mapping=False)
            if hasattr(large_fg, "fg_dict"):
                result = large_fg.find_FG(level=2)["data"]
                print("[{} 新方法识别结果]".format(smiles))
                result = large_fg.fg_dict
                large_fg.visualize_mol_with_highlight_each_fg(fg = result)
                large_fg.visualize_mol_with_highlight(fg = result)
                for key, value in result.items():
                    print(key, value)
                result = self.rd_func_group.check(smiles)
                print(result)
            else:
                print("未能识别出官能团")

    def save_csv(self):
        # 将self.df根据check_passed排序
        self.df = self.df.sort_values(by="check_passed")
        self.df.to_csv("data/Ylide/SMILES_for_FG_check-{}.csv".format(self.timestamp), index=False)

class New_Broken_Bond_Check:
    def __init__(self, project_name):
        self.project_name = project_name
        with open("metadata/project_metadata1.json", "r", encoding="utf-8") as f:
            project_metadata = json.load(f)
            project_metadata = project_metadata[project_name]

        self.timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

        self.df_target = pd.read_excel(project_metadata["target"]["rxn_data"]["path"], sheet_name=project_metadata["target"]["rxn_data"]["sheet_name"],)
        self.condition_df_target = pd.read_excel(project_metadata["target"]["condition_data"]["path"], sheet_name=project_metadata["target"]["condition_data"]["sheet_name"])

        self.df_lit = pd.read_excel(project_metadata["lit"]["rxn_data"]["path"], sheet_name=project_metadata["lit"]["rxn_data"]["sheet_name"],) 
        self.condition_df_lit = pd.read_excel(project_metadata["lit"]["condition_data"]["path"], sheet_name=project_metadata["lit"]["condition_data"]["sheet_name"])

        # self.df_target = self.df_target.reset_index()
        # self.df_lit = self.df_lit.reset_index()

        # idx赋值为df.index+1
        # self.df_target["idx"] = self.df_target.index + 1
        # self.df_lit["idx"] = self.df_lit.index + 1

        self.factory_target = Factory(
            self.df_target,
            condition_df = self.condition_df_target,
            sub_col_name_list=project_metadata["target"]["sub_col_name_list"],
            prod_col_name_list=project_metadata["target"]["prod_col_name_list"],
            reagent_col_name_list=project_metadata["target"]["reagent_col_name_list"],
            solvent_col_name_list=project_metadata["target"]["solvent_col_name_list"],
            cond_id_col_name=project_metadata["target"]["cond_id_col_name"],
            other_cond_col_name_list = project_metadata["target"]["other_cond_col_name_list"],
            freq_col_name=project_metadata["target"]["freq_col_name"],
            data_path = "data/{}".format(project_name)
        )
        self.factory_lit = Factory(
            self.df_lit,
            condition_df = self.condition_df_lit,
            sub_col_name_list=project_metadata["lit"]["sub_col_name_list"],
            prod_col_name_list=project_metadata["lit"]["prod_col_name_list"],
            reagent_col_name_list=project_metadata["lit"]["reagent_col_name_list"],
            solvent_col_name_list=project_metadata["lit"]["solvent_col_name_list"],
            cond_id_col_name=project_metadata["lit"]["cond_id_col_name"],
            other_cond_col_name_list = project_metadata["lit"]["other_cond_col_name_list"],
            freq_col_name=project_metadata["lit"]["freq_col_name"],
            data_path = "data/{}".format(project_name)
        )

        # 读入识别和统计结果
        self.factory_target.load_data_from_pickle(project_metadata["target"]["pickle_path"])
        self.factory_lit.load_data_from_pickle(project_metadata["lit"]["pickle_path"])

        # self.factory_target.get_outline_broken_formed_bond()
        # self.factory_lit.get_outline_broken_formed_bond()
        
        self.result_list = []#保存新断键信息
        self.result_dict = {}
        # 找出二级新断键
        #目的是在比较文献数据和目标数据的基础上，识别出在目标数据中未记录的化学键断裂事件，并将这些事件记录
        '''
            通过对比两数据集中 元素级断键 和 官能团级断键 的差异，
            找出参考数据集中曾出现过 而S ylide反应数据集中未出现过的断键类型，
            根据文献记录推荐相应的底物和反应条件
        '''
        for elem_key in self.factory_lit.hierarchical_broken_dict.keys():#C-C C-H N=N
            for fg_key in self.factory_lit.hierarchical_broken_dict[elem_key].keys(): #trifluoromethyl alkyl halide
                for relation in self.factory_lit.hierarchical_broken_dict[elem_key][fg_key]:#ralation 断键的类型 在不同的官能团等
                    if elem_key in self.factory_target.hierarchical_broken_dict.keys(): #lit_ele_key in target
                        if fg_key in self.factory_target.hierarchical_broken_dict[elem_key].keys():
                            if relation in self.factory_target.hierarchical_broken_dict[elem_key][fg_key]:
                                continue #lit 完全属于 target  原子 官能团 关系 都出现过了
                            else:
                                if elem_key in self.result_dict.keys():#ele_key在
                                    if fg_key in self.result_dict[elem_key].keys():#fg_key在
                                        self.result_dict[elem_key][fg_key].append(relation)
                                    else:
                                        self.result_dict[elem_key][fg_key] = [relation]#fg_key不在
                                else:
                                    self.result_dict[elem_key] = {fg_key: [relation]}#ele_key不在
                        else:
                            if elem_key in self.result_dict.keys():
                                self.result_dict[elem_key][fg_key] = [relation]
                            else:
                                self.result_dict[elem_key] = {fg_key: [relation]}
                    else:
                        if elem_key in self.result_dict.keys():
                            if fg_key in self.result_dict[elem_key].keys():
                                self.result_dict[elem_key][fg_key].append(relation)
                            else:
                                self.result_dict[elem_key][fg_key] = [relation]
                        else:
                            self.result_dict[elem_key] = {fg_key: [relation]}

        # 定义元素级断键下拉菜单
        self.drop_down_for_elem_broken_bond = widgets.Dropdown(
            options = list(self.result_dict.keys()),
            desctiption = "元素级断键",
            disabled = False,
        )
        self.drop_down_for_elem_broken_bond.observe(self.handel_elem_broken_bond_change, names='value')

        # 定义官能团级断键下拉菜单
        self.drop_down_for_fg_broken_bond = widgets.Dropdown(
            # options = list(self.factory_lit.hierarchical_broken_dict[self.drop_down_for_elem_broken_bond.value].keys()),
            desctiption = "官能团级断键",
            disabled = True,
        )
        self.drop_down_for_fg_broken_bond.observe(self.handel_fg_broken_bond_change, names='value')

        self.true_button = widgets.Button(description = "新断键符合要求")
        self.false_button = widgets.Button(description = "新断键不符合要求")
        self.output = widgets.Output()
        self.true_button.on_click(self.on_true_button_clicked)
        self.false_button.on_click(self.on_false_button_clicked)

        self.output = widgets.Output()

        display(self.drop_down_for_elem_broken_bond, self.drop_down_for_fg_broken_bond, self.true_button, self.false_button, self.output)

    def on_true_button_clicked(self, b):
        #print("on_true_button_clicked")
        with self.output:
            # 将 elem_bond, fg 存入DataFrame             for relation in self.factory_lit.hierarchical_broken_dict[elem_bond][fg]
            self.result_list.append(
                {
                    "elem_bond": self.drop_down_for_elem_broken_bond.value,
                    "fg": self.drop_down_for_fg_broken_bond.value,
                    "relation": ",".join([str(r) for r in self.factory_lit.hierarchical_broken_dict[self.drop_down_for_elem_broken_bond.value][self.drop_down_for_fg_broken_bond.value]]),
                    "TF": True
                }
            )
            self.save_result()

    def on_false_button_clicked(self, b):
        with self.output:
            self.result_list.append(
                {
                    "elem_bond": self.drop_down_for_elem_broken_bond.value,
                    "fg": self.drop_down_for_fg_broken_bond.value,
                    "relation": ",".join([str(r) for r in self.factory_lit.hierarchical_broken_dict[self.drop_down_for_elem_broken_bond.value][self.drop_down_for_fg_broken_bond.value]]),
                    "TF": False
                }
            )
            self.save_result()
    
    def save_result(self):
        # 将self.result_list转为DataFrame
        result_df = pd.DataFrame(self.result_list)
        result_df.to_csv("data/{}/New_Broken_Bond_Check-{}.csv".format(self.project_name, self.timestamp), index=False, encoding="utf_8_sig")

    def handel_elem_broken_bond_change(self, change):
        with self.output:
            # 检查S_Ylide中是否有该元素级断键
            if change.new in self.factory_target.hierarchical_broken_dict.keys():
                fg_bond_in_S_Ylide = set(list(self.factory_target.hierarchical_broken_dict[change.new].keys()))
                fg_bond_in_lit_Ylide = set(list(self.factory_lit.hierarchical_broken_dict[change.new].keys()))
                new_fg_bond = list(fg_bond_in_lit_Ylide - fg_bond_in_S_Ylide)
                print("{}中有该元素级断键，新官能团级断键有{}种".format(self.project_name, len(new_fg_bond)))
            else:
                print("{}中没有该元素级断键".format(self.project_name))
                new_fg_bond = list(self.factory_lit.hierarchical_broken_dict[change.new].keys())
            self.drop_down_for_fg_broken_bond.options = new_fg_bond
            self.drop_down_for_fg_broken_bond.disabled = False

    def handel_fg_broken_bond_change(self, change):
        with self.output:
            self.output.clear_output()
            elem_bond = self.drop_down_for_elem_broken_bond.value
            fg = change.new
            if fg in self.factory_lit.hierarchical_broken_dict[elem_bond].keys():
                for relation in self.factory_lit.hierarchical_broken_dict[elem_bond][fg]:
                    print((elem_bond, fg, relation))
                    self.factory_lit.get_comb_by_broken_fromed_bond(bond = (elem_bond, fg, relation), broken_or_formed = "broken")

class New_Broken_Bond_Check_S_Ylide:
    def __init__(self):
        self.timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.df_ylide = pd.read_excel("data/Ylide/Ylide-文献数据表.xlsx", sheet_name="文献数据表")
        self.condition_df_ylide = pd.read_excel("data/Ylide/Ylide-文献数据表.xlsx", sheet_name="条件表")

        # idx赋值为df.index+1
        self.df_ylide["idx"] = self.df_ylide.index + 1

        self.factory_ylide = Factory(self.df_ylide, condition_df = self.condition_df_ylide)

        # 读入识别和统计结果
        self.factory_ylide.load_data_from_pickle("data/Ylide/Ylide-S_Ylide-2024-12-09_18-39-30.pkl")

        self.factory_ylide.get_outline_broken_formed_bond()

        # self.result_dict = self._get_hier_broken_bond()
        self.cs_not_broken_dict, self.selected_idx_list = self._get_CS_not_broken()
        self.cs_broken_dict = self._get_CS_broken()

        self.result_dict = {}
        for elem_key in self.cs_not_broken_dict.keys():
            for fg_key in self.cs_not_broken_dict[elem_key].keys():
                for relation in self.cs_not_broken_dict[elem_key][fg_key]:
                    if elem_key in self.cs_broken_dict.keys():
                        if fg_key in self.cs_broken_dict[elem_key].keys():
                            if relation in self.cs_broken_dict[elem_key][fg_key]:
                                continue
                            else:
                                if elem_key in self.result_dict.keys():
                                    if fg_key in self.result_dict[elem_key].keys():
                                        self.result_dict[elem_key][fg_key].append(relation)
                                    else:
                                        self.result_dict[elem_key][fg_key] = [relation]
                                else:
                                    self.result_dict[elem_key] = {fg_key: [relation]}
                        else:
                            if elem_key in self.result_dict.keys():
                                self.result_dict[elem_key][fg_key] = [relation]
                            else:
                                self.result_dict[elem_key] = {fg_key: [relation]}
                    else:
                        if elem_key in self.result_dict.keys():
                            if fg_key in self.result_dict[elem_key].keys():
                                self.result_dict[elem_key][fg_key].append(relation)
                            else:
                                self.result_dict[elem_key][fg_key] = [relation]
                        else:
                            self.result_dict[elem_key] = {fg_key: [relation]}

        # 定义元素级断键下拉菜单
        self.drop_down_for_elem_broken_bond = widgets.Dropdown(
            options = list(self.result_dict.keys()),
            desctiption = "元素级断键",
            disabled = False,
        )
        self.drop_down_for_elem_broken_bond.observe(self.handel_elem_broken_bond_change, names='value')

        # 定义官能团级断键下拉菜单
        self.drop_down_for_fg_broken_bond = widgets.Dropdown(
            # options = list(self.factory_ylide_lit.hierarchical_broken_dict[self.drop_down_for_elem_broken_bond.value].keys()),
            desctiption = "官能团级断键",
            disabled = True,
        )
        self.drop_down_for_fg_broken_bond.observe(self.handel_fg_broken_bond_change, names='value')

        self.true_button = widgets.Button(description = "新断键符合要求")
        self.false_button = widgets.Button(description = "新断键不符合要求")
        self.output = widgets.Output()
        self.true_button.on_click(self.on_true_button_clicked)
        self.false_button.on_click(self.on_false_button_clicked)

        self.output = widgets.Output()

        display(self.drop_down_for_elem_broken_bond, self.drop_down_for_fg_broken_bond, self.true_button, self.false_button, self.output)

    def _get_CS_broken(self):
        result_list = []
        result_dict = {}
        for idx, data in enumerate(self.factory_ylide.data):
            if idx in self.selected_idx_list:
                continue
            if data.am_status == True and data.bond_status == True:
                parsed_broken_list = []
                selected_sub_idx = []
                for sub_idx, sub in enumerate(data.broken_bond_fg_relationships):
                    this_broken_tuple_list = []
                    flag = False
                    for broken in sub:
                        this_bond = broken["bond"]
                        if "->" in this_bond:
                            continue
                        else:
                            atom_1 = this_bond.split(" ")[0]
                            if ":" in atom_1:
                                atom_1 = this_bond.split(" ")[0].split(":")[0]
                            atom_2 = this_bond.split(" ")[2]
                            if ":" in atom_2:
                                atom_2 = this_bond.split(" ")[2].split(":")[0]
                            this_bond_type = this_bond.split(" ")[1]
                            this_bond = atom_1 + " " + this_bond_type + " " + atom_2
                        this_fg = ";".join(sorted(broken["functional_groups"]))
                        this_relationship = broken["relationship"]
                        this_broken_tuple_list.append((this_bond, this_fg, this_relationship))

                        if this_bond == "C = S":
                            flag = True
                    parsed_broken_list.append(this_broken_tuple_list)

                    if flag == False:
                        selected_sub_idx.append(sub_idx)
                for sub_idx in selected_sub_idx:
                    # 记录断键
                    result_list += parsed_broken_list[sub_idx]
                    result_list = list(set(result_list))
        for item in result_list:
            bond = item[0]
            fg = item[1]
            relation = item[2]
            if bond in result_dict.keys():
                if fg in result_dict[bond].keys():
                    if relation not in result_dict[bond][fg]:
                        result_dict[bond][fg].append(relation)
                else:
                    result_dict[bond][fg] = [relation]
            else:
                result_dict[bond] = {fg: [relation]}
        return result_dict

    def _get_CS_not_broken(self):
        idx_list = []
        for idx, data_obj in enumerate(self.factory_ylide.data):
            if data_obj.am_status == True and data_obj.bond_status == True:
                flag = False
                for changed in data_obj.all_bond_changed:
                    this_bond = changed
                    if "->" in this_bond:
                        this_bond = this_bond.split("->")[0]
                        if "C" in this_bond and "S" in this_bond and "=" in this_bond and (("Si" not in this_bond) and ("Se" not in this_bond) and ("Sn" not in this_bond) and ("Sb" not in this_bond)):
                            flag = True
                if flag == False:
                    for bond in data_obj.all_broken:
                        this_bond = bond
                        if "->" in this_bond:
                            this_bond = this_bond.split("->")[0]
                            if "C" in this_bond and "S" in this_bond and "=" in this_bond and (("Si" not in this_bond) and ("Se" not in this_bond) and ("Sn" not in this_bond) and ("Sb" not in this_bond)):
                                flag = True
                        else:
                            if "C" in this_bond and "S" in this_bond and "=" in this_bond and (("Si" not in this_bond) and ("Se" not in this_bond) and ("Sn" not in this_bond) and ("Sb" not in this_bond)):
                                flag = True
                            else:
                                continue
                if flag == False:
                    idx_list.append(idx)
                else:
                    continue
        # df = self.factory_ylide.stat_df
        # df = df[df["broken:C = S"]!=1]
        # idx_list = df["rxn_idx"].tolist()
        # print(idx_list)
        result_list = []
        result_dict = {}
        for idx in idx_list:
            data_obj = self.factory_ylide.data[idx]
            for sub_idx, sub in enumerate(data_obj.am_sub_list):
                if "S" in sub:
                    for broken in data_obj.broken_bond_fg_relationships[sub_idx]:
                        this_bond = broken["bond"]
                        if "->" in this_bond:
                            continue
                        else:
                            atom_1 = this_bond.split(" ")[0]
                            if ":" in atom_1:
                                atom_1 = this_bond.split(" ")[0].split(":")[0]
                            atom_2 = this_bond.split(" ")[2]
                            if ":" in atom_2:
                                atom_2 = this_bond.split(" ")[2].split(":")[0]
                            this_bond_type = this_bond.split(" ")[1]
                            this_bond = atom_1 + " " + this_bond_type + " " + atom_2
                        this_fg = ";".join(sorted(broken["functional_groups"]))
                        this_relationship = broken["relationship"]
                        result_list.append((this_bond, this_fg, this_relationship))
                else:
                    continue
            result_list = list(set(result_list))
        for item in result_list:
            bond = item[0]
            fg = item[1]
            relation = item[2]
            if bond in result_dict.keys():
                if fg in result_dict[bond].keys():
                    if relation not in result_dict[bond][fg]:
                        result_dict[bond][fg].append(relation)
                else:
                    result_dict[bond][fg] = [relation]
            else:
                result_dict[bond] = {fg: [relation]}
        return result_dict, idx_list

    def _get_hier_broken_bond(self):
        result_list = []
        result_dict = {}
        for idx, data in enumerate(self.factory_ylide.data):
            if data.am_status == True and data.bond_status == True:
                parsed_broken_list = []
                selected_sub_idx = []
                for sub_idx, sub in enumerate(data.broken_bond_fg_relationships):
                    this_broken_tuple_list = []
                    flag = False
                    for broken in sub:
                        this_bond = broken["bond"]
                        if "->" in this_bond:
                            continue
                        else:
                            atom_1 = this_bond.split(" ")[0]
                            if ":" in atom_1:
                                atom_1 = this_bond.split(" ")[0].split(":")[0]
                            atom_2 = this_bond.split(" ")[2]
                            if ":" in atom_2:
                                atom_2 = this_bond.split(" ")[2].split(":")[0]
                            this_bond_type = this_bond.split(" ")[1]
                            this_bond = atom_1 + " " + this_bond_type + " " + atom_2
                        this_fg = ";".join(sorted(broken["functional_groups"]))
                        this_relationship = broken["relationship"]
                        this_broken_tuple_list.append((this_bond, this_fg, this_relationship))

                        if this_bond == "C = S":
                            flag = True
                    parsed_broken_list.append(this_broken_tuple_list)

                    if flag == False:
                        selected_sub_idx.append(sub_idx)
                for sub_idx in selected_sub_idx:
                    # 记录断键
                    result_list += parsed_broken_list[sub_idx]
                    result_list = list(set(result_list))
        for item in result_list:
            bond = item[0]
            fg = item[1]
            relation = item[2]
            if bond in result_dict.keys():
                if fg in result_dict[bond].keys():
                    if relation not in result_dict[bond][fg]:
                        result_dict[bond][fg].append(relation)
                else:
                    result_dict[bond][fg] = [relation]
            else:
                result_dict[bond] = {fg: [relation]}
        return result_dict

    def on_true_button_clicked(self, b):
        with self.output:
            # 将 elem_bond, fg 存入DataFrame             for relation in self.factory_ylide_lit.hierarchical_broken_dict[elem_bond][fg]
            self.result_list.append(
                {
                    "elem_bond": self.drop_down_for_elem_broken_bond.value,
                    "fg": self.drop_down_for_fg_broken_bond.value,
                    "relation": ",".join([str(r) for r in self.factory_ylide_lit.hierarchical_broken_dict[self.drop_down_for_elem_broken_bond.value][self.drop_down_for_fg_broken_bond.value]]),
                    "TF": True
                }
            )
            self.save_result()

    def on_false_button_clicked(self, b):
        with self.output:
            self.result_list.append(
                {
                    "elem_bond": self.drop_down_for_elem_broken_bond.value,
                    "fg": self.drop_down_for_fg_broken_bond.value,
                    "relation": ",".join([str(r) for r in self.factory_ylide_lit.hierarchical_broken_dict[self.drop_down_for_elem_broken_bond.value][self.drop_down_for_fg_broken_bond.value]]),
                    "TF": False
                }
            )
            self.save_result()
    
    def save_result(self):
        # 将self.result_list转为DataFrame
        result_df = pd.DataFrame(self.result_list)
        result_df.to_csv("data/Ylide/New_Broken_Bond_Check-{}.csv".format(self.timestamp), index=False, encoding="utf_8_sig")

    def handel_elem_broken_bond_change(self, change):
        with self.output:
            new_fg_bond = list(self.result_dict[change.new].keys())
            self.drop_down_for_fg_broken_bond.options = sorted(new_fg_bond)
            self.drop_down_for_fg_broken_bond.disabled = False

    def handel_fg_broken_bond_change(self, change):
        with self.output:
            self.output.clear_output()
            elem_bond = self.drop_down_for_elem_broken_bond.value
            fg = change.new
            if fg in self.factory_ylide.hierarchical_broken_dict[elem_bond].keys():
                for relation in self.factory_ylide.hierarchical_broken_dict[elem_bond][fg]:
                    print((elem_bond, fg, relation))
                    self.factory_ylide.get_comb_by_broken_fromed_bond(bond = (elem_bond, fg, relation), broken_or_formed = "broken", sub_idx_list = self.selected_idx_list)                    