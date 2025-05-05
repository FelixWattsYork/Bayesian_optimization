from pyrokinetics import Pyro, template_dir
import os
import pathlib
from typing import Union


def main(base_path: Union[os.PathLike, str] = ".", geometry_type: str = "Miller"):
    base_path = os.getcwd()
    folder_to_convert = "GS2/Simulations/r4"
    out_loc = "TGLF/Template_files/r4"  
    base_path = pathlib.Path(base_path)
    print("got here")

    # # Equilibrium file
    # eq_file = in_loc + "/jetto.eqdsk_out"

    # # Kinetics data file
    # kinetics_file = in_loc + "/jetto.jsp"
    # print(base_path)
    # print(eq_file)

    # # Load up pyro object
    # pyro = Pyro(
    #     eq_file=eq_file,
    #     kinetics_file=kinetics_file,
    #     kinetics_type="JETTO",
    #     kinetics_kwargs={"time": 550},
    # )

    # # Generate local parameters at psi_n=0.5
    # pyro.load_local(psi_n=0.5, local_geometry=geometry_type)

    # # Write single GS2 input file, specifying the code type
    # # in the call.
    # if geometry_type == "Miller":
    #     pyro.write_gk_file(file_name=base_path / out_loc /"gs2.in", gk_code="GS2")


    # Point to GS2 input file
    gs2_template =base_path / folder_to_convert /"r4-ky1-th00-0.in"

    # Load in GS2 file
    pyro = Pyro(gk_file=gs2_template)

    # Switch to CGYRO
    pyro.gk_code = "TGLF"

    # Write CGYRO input file
    pyro.write_gk_file(file_name=base_path / out_loc /"input.tglf")

    return pyro
if __name__ == "__main__":
    main()