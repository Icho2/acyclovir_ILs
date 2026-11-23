import numpy as np
from ase.io import read
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# 1. Load single-molecule PDB templates (atom names & order must match your .xml files)
chol_pdb = app.PDBFile("choline.pdb")
ala_pdb = app.PDBFile("alaninate.pdb")
acy_pdb = app.PDBFile("acyclovir.pdb")

# 2. Build the full OpenMM Topology in exact Packmol order: (choline + alaninate) * 4 + acyclovir
topology = app.Topology()
chain = topology.addChain()

def append_residue(source_top, target_chain):
    for res in source_top.residues():
        new_res = topology.addResidue(res.name, target_chain)
        atom_map = {}
        for atom in res.atoms():
            atom_map[atom] = topology.addAtom(atom.name, atom.element, new_res)
        for bond in source_top.bonds():
            if bond[0] in atom_map and bond[1] in atom_map:
                topology.addBond(atom_map[bond[0]], atom_map[bond[1]])

for _ in range(4):
    append_residue(chol_pdb.topology, chain)
    append_residue(ala_pdb.topology, chain)
append_residue(acy_pdb.topology, chain)

# 3. Load packed coordinates from your Packmol .xyz file (Angstroms -> nm)
atoms = read("choline_alaninate_bulk_acyclovir.pdb")
positions = (atoms.get_positions() / 10.0) * unit.nanometer

# Set periodic box vectors (12 x 12 x 12 A = 1.2 x 1.2 x 1.2 nm)
topology.setPeriodicBoxVectors(np.eye(3) * 1.2 * unit.nanometer)

import io
import xml.etree.ElementTree as ET

def load_prefixed_xml(xml_path, prefix):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Rename all AtomTypes and record mapping
    type_map = {}
    atom_types = root.find("AtomTypes")
    if atom_types is not None:
        for t in atom_types.findall("Type"):
            old_name = t.attrib["name"]
            new_name = f"{prefix}_{old_name}"
            type_map[old_name] = new_name
            t.attrib["name"] = new_name

    # 2. Update type references in Residues
    residues = root.find("Residues")
    if residues is not None:
        for res in residues.findall("Residue"):
            # Ensure residue name is also unique if generic (e.g., "MOL" or "UNL")
            if res.attrib.get("name") in ("MOL", "UNL", "LIG", "UNK"):
                res.attrib["name"] = f"{prefix}_{res.attrib['name']}"
            for atom in res.findall("Atom"):
                if atom.attrib.get("type") in type_map:
                    atom.attrib["type"] = type_map[atom.attrib["type"]]

    # 3. Update type1, type2, type3, type4 across all Force sections
    for section in root:
        if section.tag in ("AtomTypes", "Residues"):
            continue
        for elem in section.iter():
            for attr in ("type", "type1", "type2", "type3", "type4"):
                if attr in elem.attrib and elem.attrib[attr] in type_map:
                    elem.attrib[attr] = type_map[elem.attrib[attr]]

    buf = io.StringIO()
    tree.write(buf, encoding="unicode")
    buf.seek(0)
    return buf

# 4. Load all three .xml force fields with unique prefixes
forcefield = app.ForceField(
    load_prefixed_xml("choline.xml", "chol"),
    load_prefixed_xml("alaninate.xml", "ala"),
    load_prefixed_xml("acyclovir.xml", "acy"),
)

# Create the OpenMM System
# Note: nonbondedCutoff must be < 0.5 * shortest box vector (0.45 nm for a 1.0 nm box)
system = forcefield.createSystem(
    topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.45 * unit.nanometer,
    constraints=app.HBonds,
)

# 5. Add Barostat (NPT at 1 bar, 300 K) and Langevin Integrator (2 fs timestep)
system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds,
)

# 6. Initialize Simulation on macOS (OpenCL or CPU)
platform = mm.Platform.getPlatformByName("OpenCL")  # Switch to "CPU" if OpenCL gives an error
simulation = app.Simulation(topology, system, integrator, platform)
simulation.context.setPositions(positions)

# Save a reference PDB of the assembled system (useful for opening the .dcd trajectory in VMD/PyMOL)
with open("system_reference.pdb", "w") as f:
    app.PDBFile.writeFile(topology, positions, f)

# 7. Minimize Energy & Equilibrate (100 ps)
print("Minimizing energy...")
simulation.minimizeEnergy(maxIterations=2000)

print("Equilibrating (100 ps NPT)...")
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
simulation.step(50_000)  # 50,000 * 2 fs = 100 ps

# 8. Production Run: 10 nanoseconds (5,000,000 steps * 2 fs)
simulation.reporters.append(app.DCDReporter("trajectory_10ns.dcd", 5000))  # Saves every 10 ps (1000 frames)
simulation.reporters.append(
    app.StateDataReporter(
        "md_log.csv",
        5000,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        density=True,
        speed=True,
    )
)

print("Running 10 ns production MD...")
simulation.step(5_000_000)
simulation.saveState("final_state.xml")
print("Simulation complete!")

