import re
from rdkit import Chem

def has_hydrogen_constraintv0(smarts: str) -> bool:
    def _atom_spec_has_h(content: str) -> bool:
        # 1. 氢原子本身
        if re.match(r'H\b|#1\b', content):
            return True

        # 先提取所有氢相关的约束（包括简写和分号）
        hydrogen_constraints = set()

        # 简写形式: [CH3], [NH2], [#6H4], [13CH3], [OH]
        simple_match = re.search(r'(?:\d*)([A-Z][a-z]?|#\d+)(?:@@?)?H([0-4]?)\b', content)
        if simple_match:
            h_count = simple_match.group(2)
            if h_count == '':
                hydrogen_constraints.add(1)   # [CH] 表示 H1
            else:
                hydrogen_constraints.add(int(h_count))

        # 分号内的 ;Hn 属性 (n=空或1-4)
        # 先找到所有 "属性列表" 部分（分号后的内容，直到下一个分号或结尾）
        # 但 SMARTS 内属性可能用逗号分隔，如 ;H1,H0
        attrs_part = content.split(';', 1)[1] if ';' in content else ''
        # 按逗号分割每个属性
        for attr in attrs_part.split(','):
            attr = attr.strip()
            # ;Hn
            h_match = re.match(r'H([0-4]?)$', attr)
            if h_match:
                val = h_match.group(1)
                if val == '':
                    hydrogen_constraints.add(1)   # H 表示 H1
                else:
                    hydrogen_constraints.add(int(val))
            # ;!H0
            if attr == '!H0':
                # 否定形式：氢数不为0，强制带氢（直接返回 True 更高效，但为了统一收集也可）
                # 直接返回 True
                return True

        # 如果未收集到任何氢约束 -> 不强制带氢（例如 [C]）
        if not hydrogen_constraints:
            return False

        # 强制带氢的条件：所有限制的氢数都 > 0
        # 注意：如果同时存在 H0 和 H1，那么氢数可以为0，整体不强制
        return all(h > 0 for h in hydrogen_constraints)
    
    bracket_pattern = r'\[([^\]]+)\]'
    
    # s = Chem.MolFromSmarts(smarts)
    # smarts = Chem.MolToSmarts(s).replace(r'&',';')
    matches = re.findall(bracket_pattern, smarts)
    if not matches:
        return False
    for content in matches:
        if _atom_spec_has_h(content):
            return True
    return False

def has_hydrogen_constraintv1(smarts: str) -> bool:
    """
    判断 SMARTS 中是否包含“必须带有氢”的约束。
    支持：;Hn (n>0), 简写 CH3, NH2 等，以及 ;!H0（非零氢）。
    """
    def _atom_spec_has_h(content: str) -> bool:
        # 1. 氢原子本身
        if re.match(r'H\b|#1\b', content):
            return True

        # 收集所有氢数限制（允许的值，如 0,1,2,3,4）
        hydrogen_allowed = set()

        # 2. 简写形式: [CH3], [NH2], [#6H4], [13CH3], [OH]
        simple_match = re.search(r'(?:\d*)([A-Z][a-z]?|#\d+)(?:@@?)?H([0-4]?)\b', content)
        if simple_match:
            h_count = simple_match.group(2)
            if h_count == '':
                hydrogen_allowed.add(1)   # [CH] 表示 H1
            else:
                hydrogen_allowed.add(int(h_count))

        # 3. 分号内的属性 — 允许属性像 X3H1 这样粘连
        content = content.replace(r'&',';')
        if ';' in content:
            # 取第一个分号之后的所有内容作为属性列表
            attrs_part = content.split(';', 1)[1]
            # 在属性字符串中查找所有 Hn / H / !H0
            # 注意：要避免把元素符号中的 H 误抓（例如 CH3 已在简写中处理，属性里不会出现 CH3）
            # 匹配模式：
            #   - H 后跟可选数字（0-4），且 H 前面不能是字母（防止匹配到 “CH3” 中的 H）
            #   - 或者 !H0
            # 使用边界确保数字不会多匹配（比如 H10 不会误判）
            for match in re.finditer(r'(?<![A-Za-z])(!?H([0-4]?))', attrs_part):
                token = match.group(1)   # 例如 "H1", "H", "!H0"
                if token == '!H0':
                    # 否定形式：氢数不能为 0，直接认为强制带氢
                    return True
                # 提取数字部分
                num_str = match.group(2)
                if num_str == '':
                    hydrogen_allowed.add(1)   # H 表示 H1
                else:
                    hydrogen_allowed.add(int(num_str))

        # 如果没有收集到任何氢约束 -> 不强制带氢
        if not hydrogen_allowed:
            return False

        # 强制带氢的条件：所有允许的氢数都大于 0
        return all(h > 0 for h in hydrogen_allowed)

    # 找出所有方括号内的原子说明符
    bracket_pattern = r'\[([^\]]+)\]'
    matches = re.findall(bracket_pattern, smarts)
    if not matches:
        return False

    for content in matches:
        if _atom_spec_has_h(content):
            return True
    return False

def has_hydrogen_constraintv2(smarts: str) -> bool:
    """
    判断 SMARTS 中是否包含“必须带有氢”的约束。
    策略：
      1. 用正则找到所有方括号内的原子说明符。
      2. 对每个原子说明符，按顶层逗号分割成多个选项（OR）。
      3. 对每个选项，使用 RDKit 将其标准化（调用 GetSmarts），从中提取氢约束。
      4. 如果原子内存在至少一个选项允许氢数为 0，则该原子不强制带氢。
      5. 只有所有选项都要求氢数 > 0（或含有 !H0），该原子才算强制带氢。
      6. 任意一个原子强制带氢即整体返回 True。
    """
    def option_has_required_h(option: str) -> bool:
        """判断一个原子选项（不含外层方括号，且不含顶层逗号）是否强制要求氢数 > 0。"""
        # 使用 RDKit 解析原子，获取标准化 SMARTS
        mol = Chem.MolFromSmarts(f"[{option}]")
        if mol is None or mol.GetNumAtoms() == 0:
            # 无法解析，回退到直接字符串判断（兼容性）
            return _legacy_has_h(option)
        atom = mol.GetAtomWithIdx(0)
        std = atom.GetSmarts()   # 例如 "[#6&H3]", "[#7&!H0]"
        inner = std[1:-1]

        # 检查否定形式 &!H0
        if "&!H0" in inner:
            return True

        # 检查 &H 或 &Hn
        match = re.search(r'&H([0-4]?)', inner)
        if match:
            val = match.group(1)
            if val == '' or int(val) > 0:
                return True
        return False

    def _legacy_has_h(option: str) -> bool:
        """当 RDKit 无法解析时，使用简单的字符串匹配（兼容极少数情况）。"""
        # 判断氢原子本身
        if re.match(r'H\b|#1\b', option):
            return True
        # 查找 !H0, H, H1-H4（包括粘连）
        return bool(re.search(r'!H0|(?<![A-Za-z])H([0-4]?)', option))

    # 找出所有方括号内的原子说明符
    bracket_pattern = r'\[([^\]]+)\]'
    matches = re.findall(bracket_pattern, smarts)
    if not matches:
        return False

    for atom_content in matches:
        # 分割顶层逗号（忽略括号内的逗号）
        options = []
        depth = 0
        start = 0
        for i, ch in enumerate(atom_content):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
            elif ch == ',' and depth == 0:
                options.append(atom_content[start:i])
                start = i + 1
        options.append(atom_content[start:])

        # 判断该原子的所有选项是否都强制带氢
        all_require = True
        for opt in options:
            opt = opt.strip()
            if not option_has_required_h(opt):
                # 存在一个选项允许零氢 → 该原子整体不强制
                all_require = False
                break
        if all_require:
            return True

    return False


def has_hydrogen_constraint(smarts: str) -> bool:
    """
    判断 SMARTS 中是否包含“必须带有氢”的约束。
    支持：
      - 简写 [CH3], [NH2], [OH], [#6H4]
      - 分号属性 ;Hn, ;!H0, 以及粘连形式 X3H1
      - 逗号分隔的 OR 选项（如 ;H1,H0）
      - 递归 SMARTS $()（不作为氢约束）
    使用 RDKit 标准化原子属性，但手动处理 OR 语义。
    """
    
    def option_has_required_h(option: str) -> bool:
        """
        判断一个原子选项（不含外层方括号，且本身不是递归 SMARTS）
        是否强制要求氢数 > 0。
        """
        # 如果是递归 SMARTS，本身不对当前原子施加氢约束
        if option.startswith('$('):
            return False
        
        # 使用 RDKit 解析该原子选项，获取标准化 SMARTS
        mol = Chem.MolFromSmarts(f'[{option}]')
        if mol is None or mol.GetNumAtoms() == 0:
            # 回退：简单正则匹配
            return _legacy_has_h(option)
        
        atom = mol.GetAtomWithIdx(0)
        std = atom.GetSmarts()          # 如 "[#6&H3]"
        inner = std[1:-1]
        
        # 否定形式 &!H0
        if '&!H0' in inner:
            return True
        # &H 或 &Hn
        match = re.search(r'&H([0-4]?)', inner)
        if match:
            val = match.group(1)
            if val == '' or int(val) > 0:
                return True
        return False
    
    def _legacy_has_h(option: str) -> bool:
        """极少数无法被 RDKit 解析时的回退方案"""
        if re.match(r'H\b|#1\b', option):
            return True
        return bool(re.search(r'!H0|(?<![A-Za-z])H([0-4]?)', option))
    
    # 找出所有方括号内的原子说明符
    bracket_pattern = r'\[([^\]]+)\]'
    matches = re.findall(bracket_pattern, smarts)
    if not matches:
        return False
    
    for atom_content in matches:
        if '$' in atom_content:
            continue
        # 分割顶层逗号（忽略括号内的逗号，包括递归 SMARTS 的括号）
        options = []
        depth = 0
        start = 0
        i = 0
        while i < len(atom_content):
            ch = atom_content[i]
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
            elif ch == '(' and atom_content[i-1:i+1] != '$(':   # 避免将 $ 后的括号算作普通括号？其实我们只需计数所有圆括号
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                options.append(atom_content[start:i])
                start = i + 1
            i += 1
        options.append(atom_content[start:])
        
        # 判断该原子的所有选项是否都强制带氢
        all_require = True
        for opt in options:
            opt = opt.strip()
            
            if not option_has_required_h(opt):
                all_require = False
                break
        if all_require:
            return True
    
    return False
def create_fgs_with_ele(smarts):
    res = []
    s = Chem.MolFromSmarts(smarts)
    def add_atom(atom):
        atom.SetAtomMapNum(3)
        res.append(Chem.MolToSmarts(s))
        atom.SetAtomMapNum(0)
    for atom in s.GetAtoms():
        a_sbol = atom.GetSymbol() 
        a_char = atom.GetFormalCharge()
        if a_sbol in ['N', 'O', 'S', 'P', 'As', 'Se']:
            # if smarts not in ['[#7;X3H0]([#6])([#6])([#6])']:
            if has_hydrogen_constraint(atom.GetSmarts()): #排除不包含除氢的
                continue
            
            bonded_count = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())  + a_char
            if a_sbol in ['N', 'P']: 
                if bonded_count >= 4:# in [2,3] or bonded_count==4.5: #[#7+]
                    continue
                add_atom(atom)
            elif a_sbol == 'S':
                if bonded_count == 6 or 'v6' in atom.GetSmarts():
                    continue            
                # if bonded_count in [2, 4] or 'v4' in atom.GetSmarts(): #[#7+]
                #     add_atom(atom)
                # elif bonded_count == 3 and atom.GetIsAromatic():
                add_atom(atom)
            elif a_sbol == 'O': #and a_char == -1:
                #if bonded_count == 0: #[#7+][O-]
                add_atom(atom)
            elif a_sbol == 'As':
                add_atom(atom)
            elif a_sbol == 'Se':
                add_atom(atom)
    return res

def test_hydrogen_constraint():
    # 应返回 True 的示例
    assert has_hydrogen_constraint("[CH3]") == True
    assert has_hydrogen_constraint("[#6]1([#6]2)[#6][#6]([#6][#6]2[#7;H1]3)[#6][#6]3[#6]1") == True
    assert has_hydrogen_constraint("[C;H3]") == True
    assert has_hydrogen_constraint("[NH2]") == True
    assert has_hydrogen_constraint("[#6H4]") == True
    assert has_hydrogen_constraint("[13CH3]") == True
    assert has_hydrogen_constraint("[OH]") == True
    assert has_hydrogen_constraint("[C;H1]") == True
    assert has_hydrogen_constraint("[N;!H0]") == True
    assert has_hydrogen_constraint("[#7;H1]") == True
    assert has_hydrogen_constraint("[#7H]") == True
    assert has_hydrogen_constraint("[#7;X3H1]([#6])([#6])") == True
    assert has_hydrogen_constraint("[#7;X3H1]") == True
    assert has_hydrogen_constraint("[#8X2H]") == True
    assert has_hydrogen_constraint("[#8;H1,$([#8]([#6])[#6])][#6](=[#8])[#8][#6]]") == False


    # 应返回 False 的示例（无氢约束或明确 H0）
    assert has_hydrogen_constraint('[#7;X3H0]([#6])([#6])([#6])') == False
    assert has_hydrogen_constraint("[C]") == False
    assert has_hydrogen_constraint("[#7;X3H0]") == False   # 你给出的例子
    assert has_hydrogen_constraint("[C;H0]") == False
    assert has_hydrogen_constraint("[#6]") == False
    assert has_hydrogen_constraint("O([!H])") == False
    assert has_hydrogen_constraint("O([!H0])") == False
    assert has_hydrogen_constraint("c1ccc([I]([!H])c2ccccc2)cc1") == False
    assert has_hydrogen_constraint("[#7;!a;X4H+,X4H2+,X4H3+,X3H+0,X3H0+0,X3H2+0;!$([#7][!#6])]") == False
    assert has_hydrogen_constraint("[#8]=[#6](-[#6])-[#8]-[I]([!H])-[c]1[c][c][c][c][c]1") == False
    assert has_hydrogen_constraint("[#16v6](=[#8])(=[#8])[#7;H1,H0][#7;X3H1+0,X3H2+0,X3H0+0]") == False
    assert has_hydrogen_constraint("[#16v6](=[#8])(=[#8])[#7;H1,H0][#7;X3H1+0,X3H2+0,X3H0+0]") == False
    assert has_hydrogen_constraint("[#7;H1,H0][#7;X3H1+0,X3H2+0,X3H0+0]") == False
    assert has_hydrogen_constraint("[#6;H0,H1,H2]([#7;X3H1+0,X3H2+0,X3H0+0])") == False
    assert has_hydrogen_constraint("[#7X2;H0]=[#6]=[#8]") == False

    assert has_hydrogen_constraint("[#7X2;H0]=[#6]=[#8]") == False

    assert has_hydrogen_constraint("[n]1[c][se][c][n]1") == False
    assert has_hydrogen_constraint("[c]1[c][c][c][se]1") == False

def test_create_fgs_with_ele():
    ls = ['[#6][As]([#6])=[#8]', '[#6][As][#6]','[#6][As]([#6])=[#8]',
          '[#6;H1,H2](=[#16X1])', '[#16v2][#6;H3]',
          '[s]1[c][c][c][p]1', '[s]1[n][p][n][s][n]1', '[P]',
          '[n]1[c][n-][c][c]1','[n+]','[#6][#7+]#[#6-]','[#7;X3H0]([#6])([#6])([#6])','[#7;X3H1]([#6])([#6])','[#7;X3H2][#6]',
          '[#8X2H]','[#5X3]([#8;H])([#8;H])[#6]','[#6](=[#8])[#8X2H]','[#8;H1,$([#8]([#6])[#6])][#6](=[#8])[#8][#6]',
          '[#16v6](=[#8])(=[#8])[#7;H1,H0]']
    for s in ls[-1:]:
        res = create_fgs_with_ele(s)
        print(res)



if __name__ == '__main__':
    test_hydrogen_constraint()
    test_create_fgs_with_ele()