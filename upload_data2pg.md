# Copy smiles data to postgresql 
In postgresql database:

```SQL
CREATE DATABASE bide_DB;
CREATE EXTENSION rdkit;
CREATE SCHEMA dpph;

CREATE TABLE dpph.avail_smiles(smiles TEXT,cansmiles TEXT, price FLOAT, source TEXT, mol mol, ac SMALLINT, amw FLOAT);
CREATE TABLE dpph.avail_smiles_1044(smiles TEXT, ac SMALLINT, amw FLOAT, mol mol, is_463 SMALLINT);

-- copy raw data to sql
\copy dpph.avail_smiles(smiles, cansmiles, price, source,  ac, amw) FROM 'PATH/data/avail_smiles_3w2' CSV HEADER;
ALTER TABLE dpph.avail_smiles SET mol = mol_from_smiles(smiles::cstring);
\copy dpph.avail_smiles(smiles,  mol, ac, amw) FROM 'PATH/data/avail_smiles_1044.sv' CSV HEADER;
ALTER TABLE dpph.avail_smiles_1044 SET mol = mol_from_smiles(smiles::cstring);

-- create index
CREATE INDEX mol_index ON dpph.avail_smiles USING gist(mol);
CREATE INDEX mol_index_1044 ON dpph.avail_smiles_1044 USING gist(mol);
```