from pyrokinetics import Pyro, PyroScan
import os
import pathlib
from typing import Union
import numpy as np


def main(base_path: Union[os.PathLike, str] = ".", geometry_type: str = "Miller"):
    base_path = os.getcwd()
    folder_to_convert = "GS2/Simulations/r1"
    out_loc = "GS2/Simulations/r1_scan"  
    base_path = pathlib.Path(base_path)
    print("got here")

    # Point to GS2 input file
    gs2_template =base_path / folder_to_convert /"r1-ky1-th00-0.in"

    # Load in GS2 file
    pyro = Pyro(gk_file=gs2_template)

     # Write input files
    param_dict = {"ky": np.arange(0, 1, 0.01)}

    # Create PyroScan object
    pyro_scan = PyroScan(
    pyro,
    param_dict,
    base_directory="run_directory"
    )

    pyro_scan.write(file_name="gs2.in", base_directory=out_loc)

    return pyro
if __name__ == "__main__":
    main()