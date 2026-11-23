from pathlib import Path
import mdtraj as md
from ase import Atoms
from ase.io import Trajectory


def dcd_to_ase_traj_mdtraj(
    dcd_path: Path,
    topology_path: Path,
    traj_path: Path,
    stride: int = 1,
) -> None:
    traj_md = md.load(str(dcd_path), top=str(topology_path), stride=stride)
    symbols = [atom.element.symbol for atom in traj_md.topology.atoms]
    coords_ang = traj_md.xyz * 10.0  # nm -> Angstrom

    with Trajectory(str(traj_path), mode="w") as traj_ase:
        for i in range(traj_md.n_frames):
            cell = None
            pbc = False
            if traj_md.unitcell_vectors is not None and traj_md.unitcell_vectors[i] is not None:
                # MDTraj unitcell_vectors are in nm -> convert to Angstrom
                cell = traj_md.unitcell_vectors[i] * 10.0
                pbc = True

            atoms = Atoms(symbols=symbols, positions=coords_ang[i], cell=cell, pbc=pbc)
            traj_ase.write(atoms)


if __name__ == "__main__":
    dcd_to_ase_traj_mdtraj(
        dcd_path=Path("trajectory_10ns.dcd"),
        topology_path=Path("system_reference.pdb"),
        traj_path=Path("output.traj"),
        stride=1,
    )
