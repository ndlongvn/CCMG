"""SMILES normalization and deterministic BRICS/ring motif decomposition."""

from rdkit import Chem
from rdkit.Chem import BRICS


def neutralize_atoms(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize eligible atoms in place using the RDKit Cookbook pattern.

    Reference: https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    for (atom_index,) in mol.GetSubstructMatches(pattern):
        atom = mol.GetAtomWithIdx(atom_index)
        charge = atom.GetFormalCharge()
        hydrogen_count = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hydrogen_count - charge)
        atom.UpdatePropertyCache()
    return mol


def remove_salts_solvents(smiles: str) -> str | None:
    """Keep the fragment with the most heavy atoms, breaking ties by input order.

    Removed fragments are not necessarily actual salts or solvents. Return None
    if a fragment is invalid. Single-fragment strings are returned unchanged.
    """
    fragments = smiles.split(".")
    if len(fragments) == 1:
        return smiles
    molecules = [Chem.MolFromSmiles(fragment) for fragment in fragments]
    if any(molecule is None for molecule in molecules):
        return None
    largest = max(molecules, key=lambda molecule: molecule.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest)


def wash_smiles(smiles: str) -> str | None:
    """Return canonical, neutralized SMILES or None for invalid/empty inputs."""
    if not isinstance(smiles, str) or not smiles:
        return None
    smiles = remove_salts_solvents(smiles)
    if smiles is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(neutralize_atoms(molecule), isomericSmiles=True)


def _brics_edges(mol: Chem.Mol) -> set[tuple[int, int]]:
    """Return atom pairs BRICS would cut, ordered within each pair.

    FindBRICSBonds yields ((atom_i, atom_j), (label_i, label_j)).
    """
    cuts = set()
    for atom_pair, _ in BRICS.FindBRICSBonds(mol):
        i, j = map(int, atom_pair)
        if i != j:
            cuts.add((min(i, j), max(i, j)))
    return cuts


def _components_without_edges(
    mol: Chem.Mol, cut_edges: set[tuple[int, int]]
) -> list[set[int]]:
    """Find connected components after removing the specified atom-pair edges."""
    neighbors = {i: set() for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (min(i, j), max(i, j)) not in cut_edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
    components, seen = [], set()
    for atom in neighbors:
        if atom in seen:
            continue
        stack, component = [atom], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(neighbors[current] - seen)
        components.append(component)
    return components


def motif_decomp_fixed(
    mol, drop_full_molecule: bool = True, brics_only: bool = False
) -> list[list[int]]:
    """Decompose a molecule into BRICS components and optional individual rings.

    Components containing only rings emit their individual SSSR rings; mixed
    ring/linker components stay intact. Motifs may overlap but must cover every
    atom. Return an empty list when no valid full cover remains after dropping
    whole-molecule motifs. Single-atom molecules retain their singleton motif.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 1:
        return [[0]]

    components = _components_without_edges(mol, _brics_edges(mol))
    rings = [set(map(int, ring)) for ring in Chem.GetSymmSSSR(mol)]
    motifs = []
    for component in components:
        if brics_only:
            motifs.append(sorted(component))
            continue
        contained_rings = [ring for ring in rings if ring <= component]
        used = set().union(*contained_rings) if contained_rings else set()
        if not used or component - used:
            motifs.append(sorted(component))
        else:
            motifs.extend(
                sorted(ring) for ring in contained_rings if 0 < len(ring) < num_atoms
            )

    if drop_full_molecule:
        motifs = [motif for motif in motifs if 0 < len(motif) < num_atoms]
    else:
        motifs = [motif for motif in motifs if motif]
    covered = set().union(*map(set, motifs)) if motifs else set()
    if covered != set(range(num_atoms)):
        return []
    return sorted(motifs, key=lambda motif: (len(motif), motif))
