from rdkit import Chem

# 含孤对电子的原子价电子数: N(VA)→5, O(VIA)→6, P(VA)→5, S(VIA)→6, As(VA)→5, Se(VIA)→6
VALENCE_E = {7: 5, 8: 6, 15: 5, 16: 6, 33: 5, 34: 6}


def count_lone_pairs(atom):
    """计算原子上实际孤对电子数（考虑成键环境和形式电荷）"""
    an = atom.GetAtomicNum()
    if an not in VALENCE_E:
        return 0
    n_unpaired = VALENCE_E[an] - atom.GetFormalCharge() - atom.GetTotalValence()
    return max(0, n_unpaired // 2)


def match_with_lone_pair(mol, smarts, map_tag=3):
    """
    匹配 SMARTS，仅返回 :<map_tag> 标记的原子实际有孤对电子的匹配结果
    """
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return []

    # 找到 pattern 中 map_tag 对应的原子索引
    map_idx = None
    for a in pat.GetAtoms():
        if a.GetAtomMapNum() == map_tag:
            map_idx = a.GetIdx()
            break

    matches = mol.GetSubstructMatches(pat)
    if map_idx is None or not matches:
        return matches

    return [m for m in matches if count_lone_pairs(mol.GetAtomWithIdx(m[map_idx])) > 0]


if __name__ == "__main__":
    SMILES = ['O=C(c1cc(Br)ccc1)NSc2ccccc2','SC1=NC(C2=CC=NC=C2)=CS1','c1cccnc1']
    s = ['[#8:3]', '[#6](=[#8:3])[#7]','[#16:3]']

    # 测试验证
    test_mols = {
        '三乙胺': 'CCN(CC)CC',
        '水': 'O',
        '二甲基硫': 'CSC',
        '二甲基亚砜': 'CS(=O)C',
        '二甲基砜': 'CS(=O)(=O)C',
    }

    print("=== 孤对电子检测验证 ===")
    for name, smi in test_mols.items():
        mol = Chem.MolFromSmiles(smi)
        for a in mol.GetAtoms():
            lp = count_lone_pairs(a)
            if a.GetAtomicNum() in VALENCE_E.keys():
                print(f"  {name} - {a.GetSymbol()} (idx={a.GetIdx()}): {lp} 对孤对电子")

    print("\n=== SMARTS 匹配（含孤对电子约束）===")
    for smi in SMILES:
        mol = Chem.MolFromSmiles(smi)
        print(f"\n{smi}")
        for smarts in s:
            matches = match_with_lone_pair(mol, smarts)
            if matches:
                atoms_desc = []
                for m in matches:
                    map3_idx_in_mol = -1
                    pat = Chem.MolFromSmarts(smarts)
                    for a in pat.GetAtoms():
                        if a.GetAtomMapNum() == 3:
                            map3_idx_in_mol = a.GetIdx()
                            break
                    if map3_idx_in_mol >= 0:
                        a = mol.GetAtomWithIdx(m[map3_idx_in_mol])
                        atoms_desc.append(f"  match={m}, :3 原子 = {a.GetSymbol()}(idx={m[map3_idx_in_mol]})")
                print(f"  [{smarts}]")
                for d in atoms_desc:
                    print(d)

