# -*- coding: utf-8 -*-
import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import pandas as pd

FOCUS_ELEMENTS = {7, 8, 15, 16, 33, 34} # N O P S As Se
VALENCE_ELECTRONS = {7: 5, 8: 6, 15: 5, 16: 6, 33: 5, 34: 6}


class LonePairAnalyzer:
    """分析反应中 N/O/S 的孤对电子断键 / 成键"""

    def __init__(self, rxn_smiles):
        self.rxn_smiles = rxn_smiles
        self._parse()
        self._build_bond_sets()
        self._analyze()

    # ── 解析 ──────────────────────────────────

    def _parse(self):
        rxn = AllChem.ReactionFromSmarts(self.rxn_smiles, useSmiles=True)
        if rxn is None:
            raise ValueError("无法解析反应SMILES")

        reactants, products = rxn.GetReactants(), rxn.GetProducts()

        # Save per-reactant atom maps for 2D output
        self._reactant_maps = []
        for mol in reactants:
            maps = set()
            for atom in mol.GetAtoms():
                m = atom.GetAtomMapNum()
                if m > 0:
                    maps.add(m)
            self._reactant_maps.append(maps)

        def _combine(mols):
            mol = mols[0]
            for i in range(1, len(mols)):
                mol = rdmolops.CombineMols(mol, mols[i])
            return mol

        self._r_mol = reactants[0] if len(reactants) == 1 else _combine(reactants)
        self._p_mol = products[0] if len(products) == 1 else _combine(products)
        for mol in (self._r_mol, self._p_mol):
            Chem.SanitizeMol(mol)

    # ── 原子属性 ──────────────────────────────

    @staticmethod
    def _lone_pairs(atom):
        ve = VALENCE_ELECTRONS.get(atom.GetAtomicNum())
        if ve is None:
            return None
        lone_e = ve - atom.GetFormalCharge() - atom.GetTotalValence()
        return max(0, lone_e) // 2

    def _get_props(self, mol):
        props = {}
        for atom in mol.GetAtoms():
            m = atom.GetAtomMapNum()
            if m > 0:
                props[m] = {
                    'fc': atom.GetFormalCharge(),
                    'tv': atom.GetTotalValence(),
                    'degree': atom.GetDegree(),
                    'anum': atom.GetAtomicNum(),
                    'sym': atom.GetSymbol(),
                    'lp': self._lone_pairs(atom),
                }
        return props

    # ── 键集合 ────────────────────────────────

    @staticmethod
    def _get_bonds(mol):
        bonds = set()
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            if a1 > 0 and a2 > 0:
                if a1 > a2:
                    a1, a2 = a2, a1
                bonds.add((a1, a2))
        return bonds

    def _build_bond_sets(self):
        self._r_props = self._get_props(self._r_mol)
        self._p_props = self._get_props(self._p_mol)
        r_bonds = self._get_bonds(self._r_mol)
        p_bonds = self._get_bonds(self._p_mol)
        self._formed = p_bonds - r_bonds
        self._broken = r_bonds - p_bonds
        self._all_maps = sorted(set(self._r_props) | set(self._p_props))

    # ── 辅助 ──────────────────────────────────

    def _dlp(self, m):
        """LP 变化量 (产物 − 反应物), 不可算时返回 0"""
        p = self._r_props.get(m)
        q = self._p_props.get(m)
        if p is None or q is None:
            return 0
        if p['lp'] is None or q['lp'] is None:
            return 0
        return q['lp'] - p['lp']

    def _bond_str(self, a1, a2):
        """'X:m - Y:n' 形式"""
        src = self._p_props if a1 in self._p_props else self._r_props
        dst = self._p_props if a2 in self._p_props else self._r_props
        return f"{src[a1]['sym']}:{a1} - {dst[a2]['sym']}:{a2}"

    def _ele_str(self, m):
        """'X:m - E' 形式 (孤对电子事件)"""
        src = self._r_props if m in self._r_props else self._p_props
        return f"{src[m]['sym']}:{m} - E"

    def _is_focus(self, anum):
        return anum in FOCUS_ELEMENTS

    # ── 核心分析 ──────────────────────────────

    def _analyze(self,only_consider_E=True):
        self.broken = []
        self.formed = []

        # 1) 孤对电子增减事件
        for m in self._all_maps:
            d = self._dlp(m)
            if d < 0:
                self.broken.append(self._ele_str(m))
            elif d > 0:
                self.formed.append(self._ele_str(m))

        # 2) 键断裂 (涉及 N/O/S 的异裂 / 离去基团)
        for a1, a2 in sorted(self._broken):
            if not (a1 in self._r_props and a2 in self._r_props):
                continue
            an1, an2 = self._r_props[a1]['anum'], self._r_props[a2]['anum']
            if not (self._is_focus(an1) or self._is_focus(an2)):
                continue

            dlp1, dlp2 = self._dlp(a1), self._dlp(a2)
            in_p1, in_p2 = a1 in self._p_props, a2 in self._p_props
            hit = False

            if dlp1 > 0 or dlp2 > 0:          # N/O/S 获得孤对
                hit = True
            elif in_p1 != in_p2:               # 离去基团
                leaving = a2 if in_p1 else a1
                if self._is_focus(self._r_props[leaving]['anum']):
                    hit = True
            elif not in_p1 and not in_p2:      # 离去基团内部键
                continue

            if hit:
                self.broken.append(self._bond_str(a1, a2))

        # 3) 键生成 (N/O/S 贡献孤对)
        for a1, a2 in sorted(self._formed):
            if not (a1 in self._p_props and a2 in self._p_props):
                continue
            an1, an2 = self._p_props[a1]['anum'], self._p_props[a2]['anum']
            if not (self._is_focus(an1) or self._is_focus(an2)):
                continue
            if self._dlp(a1) < 0 or self._dlp(a2) < 0:
                self.formed.append(self._bond_str(a1, a2))
        if only_consider_E:
            self.formed = [s for s in self.formed if s.split('-')[1].strip() == 'E']
            self.broken = [s for s in self.broken if s.split('-')[1].strip() == 'E']

    # ── 外部接口 ──────────────────────────────

    @property
    def result(self):
        # 按反应物碎片组织为二维列表
        broken_2d = [[] for _ in self._reactant_maps]
        formed_2d = [[] for _ in self._reactant_maps]

        def _bucket(items, buckets):
            for item in items:
                m = int(item.split()[0].split(':')[1])
                for i, maps in enumerate(self._reactant_maps):
                    if m in maps:
                        buckets[i].append(item)
                        break

        _bucket(self.broken, broken_2d)
        _bucket(self.formed, formed_2d)
        return broken_2d, formed_2d


if __name__ == '__main__':
    df = pd.read_excel('data/element_bf_checkout.xlsx')

    for i, (idx, row) in enumerate(df.iterrows()):
        rxn = row['smiles_am']
        if "None" not in df.loc[i, 'broken_each_reactant_list']:
            continue
        flag = 0
        for x in ['Cl','Li','Na','K']:
            if x in rxn:
                flag = 1
                break
        if flag:
            continue
        try:
            analyzer = LonePairAnalyzer(rxn)
            broken, formed = analyzer.result
            print(f"# {row['id']}")
            print(f"断键: {broken}")
            print(f"成键: {formed}")
            print()
        except Exception as e:
            print(f"# {row.get('id', i)} 错误: {e}")
            print()
