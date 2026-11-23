import io
import xml.etree.ElementTree as ET
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# 1. Load single-molecule PDB templates
chol_pdb = app.PDBFile("choline.pdb")
ala_pdb = app.PDBFile("alaninate.pdb")
acy_pdb = app.PDBFile("acyclovir.pdb")

# 2. Build Topology in exact Packmol order: 30 choline -> 30 alaninate -> 15 acyclovir
topology = app.Topology()
chain = topology.addChain()

def append_molecules(source_top, target_chain, count):
    for _ in range(count):
        for res in source_top.residues():
            new_res = topology.addResidue(res.name, target_chain)
            atom_map = {}
            for atom in res.atoms():
                atom_map[atom] = topology.addAtom(atom.name, atom.element, new_res)
            for bond in source_top.bonds():
                if bond[0] in atom_map and bond[1] in atom_map:
                    topology.addBond(atom_map[bond[0]], atom_map[bond[1]])

append_molecules(chol_pdb.topology, chain, 100)
append_molecules(ala_pdb.topology, chain, 100)
append_molecules(acy_pdb.topology, chain, 50)

# 3. Load packed coordinates directly with OpenMM (automatically converts Angstroms -> nm)
packed_pdb = app.PDBFile("choline_alaninate_bulk_acyclovir.pdb")
positions = packed_pdb.positions

# Set periodic box vectors (50 x 50 x 50 A = 5.0 x 5.0 x 5.0 nm)
topology.setPeriodicBoxVectors(np.eye(3) * 5.0 * unit.nanometer)

def load_prefixed_xml(xml_path, prefix, charge_scale=1.0):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Rename AtomTypes AND classes so LigParGen C800/N800 names never collide
    type_map = {}
    class_map = {}
    atom_types = root.find("AtomTypes")
    if atom_types is not None:
        for t in atom_types.findall("Type"):
            old_type = t.attrib["name"]
            new_type = f"{prefix}_{old_type}"
            type_map[old_type] = new_type
            t.attrib["name"] = new_type

            if "class" in t.attrib:
                old_class = t.attrib["class"]
                new_class = f"{prefix}_{old_class}"
                class_map[old_class] = new_class
                t.attrib["class"] = new_class

    # Update type references in Residues
    residues = root.find("Residues")
    if residues is not None:
        for res in residues.findall("Residue"):
            if res.attrib.get("name") in ("MOL", "UNL", "LIG", "UNK"):
                res.attrib["name"] = f"{prefix}_{res.attrib['name']}"
            for atom in res.findall("Atom"):
                if atom.attrib.get("type") in type_map:
                    atom.attrib["type"] = type_map[atom.attrib["type"]]

    # Update type* and class* attributes across all Force sections & scale charges
    for section in root:
        if section.tag in ("AtomTypes", "Residues"):
            continue
        for elem in section.iter():
            for attr in ("type", "type1", "type2", "type3", "type4"):
                if attr in elem.attrib and elem.attrib[attr] in type_map:
                    elem.attrib[attr] = type_map[elem.attrib[attr]]
            for attr in ("class", "class1", "class2", "class3", "class4"):
                if attr in elem.attrib and elem.attrib[attr] in class_map:
                    elem.attrib[attr] = class_map[elem.attrib[attr]]
            if section.tag == "NonbondedForce" and "charge" in elem.attrib:
                elem.attrib["charge"] = f"{float(elem.attrib['charge']) * charge_scale:.6f}"

    buf = io.StringIO()
    tree.write(buf, encoding="unicode")
    buf.seek(0)
    return buf

# 4. Load force fields (0.8 charge scaling on ions; 1.0 on neutral acyclovir)
forcefield = app.ForceField(
    load_prefixed_xml("choline.xml", "chol", charge_scale=0.8),
    load_prefixed_xml("alaninate.xml", "ala", charge_scale=0.8),
    load_prefixed_xml("acyclovir.xml", "acy", charge_scale=1.0),
)

# 5. Create OpenMM System with 1.0 nm cutoff for the 3.0 nm box
system = forcefield.createSystem(
    topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)

# 6. Add Barostat (NPT at 1 bar, 300 K) and Langevin Integrator (2 fs timestep)
system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds,
)

# 7. Initialize Simulation on macOS (OpenCL or CPU)
platform = mm.Platform.getPlatformByName("OpenCL")  # Switch to "CPU" if OpenCL gives an error
simulation = app.Simulation(topology, system, integrator, platform)
simulation.context.setPositions(positions)

# Save a reference PDB of the assembled system (useful for opening the .dcd trajectory in VMD/PyMOL)
with open("system_reference.pdb", "w") as f:
    app.PDBFile.writeFile(topology, positions, f)

# 8. Minimize Energy & Equilibrate (100 ps)
print("Minimizing energy...")
simulation.minimizeEnergy(maxIterations=2000)

print("Equilibrating (100 ps NPT)...")
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
simulation.step(50_000)  # 50,000 * 2 fs = 100 ps

# 9. Production Run: 10 nanoseconds (5,000,000 steps * 2 fs)
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
