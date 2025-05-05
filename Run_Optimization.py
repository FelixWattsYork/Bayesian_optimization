import warnings
import numpy as np
import xarray as xr
import subprocess
import os
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from sklearn.gaussian_process.kernels import Matern
from matplotlib.backends.backend_pdf import PdfPages
from pyrokinetics import Pyro
import base64
from io import BytesIO

# suppress warnings about this being an experimental feature
warnings.filterwarnings(action="ignore")

simulation_list = ["SPR-045"]
KY_RANGE = np.arange(0.01, 1, 0.01)

TGLF_FIles_Directory = "/home/felix/Documents/Physics_Work/Project_Codes/Bayesian_Files/TGLF/Simulations"
GS2_Files_Directory = "/home/felix/Documents/Physics_Work/Project_Codes/Bayesian_Files/TGLF/Simulations"


def generate_tglf_files(template_path, params_to_update, ky_range):
    """
    Generate TGLF input files with updated parameters and a KY parameter scan.

    Args:
        template_path (str): Path to the template TGLF file.
        output_base_dir (str): Base directory to save the generated files.
        params_to_update (dict): Dictionary of parameters to update with their new values.
        ky_range (list): List of KY values to scan.
    """
    output_base_dir = os.path.join("TGLF/Simulations",template_path)
    templ_loc = os.path.join(os.getcwd(),"TGLF/Template_files", template_path, "input.tglf")
    if not os.path.exists(templ_loc):
        raise FileNotFoundError(f"Template file not found: {templ_loc}") 

    with open(templ_loc, "r") as template_file:
        template_lines = template_file.readlines()

    param_dir = "_".join([f"{key}{value}" for key, value in params_to_update.items() if key != "KY"])
    os.makedirs(param_dir, exist_ok=True)

    for ky in ky_range:
        # Update KY value for this scan
        params_to_update["KY"] = ky

        # Create a directory name with key parameters
        ky_dir = os.path.join(output_base_dir,param_dir, f"ky_{ky:.2f}")
        os.makedirs(ky_dir, exist_ok=True)

        # Update the template file with the new parameters
        updated_lines = []
        for line in template_lines:
            key_value = line.split("=")
            if len(key_value) == 2:
                key = key_value[0].strip()
                if key in params_to_update:
                    updated_lines.append(f"{key} = {params_to_update[key]}\n")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        # Write the updated file to the KY-specific directory
        output_path = os.path.join(ky_dir, "input.tglf")
        with open(output_path, "w") as output_file:
            output_file.writelines(updated_lines)

def run_tglf_on_files(simulation,params_to_update):
    """
    Run TGLF on all input files in the specified directory.

    Args:
        directory (str): Path to the directory containing input files.
    """

    param_dir = "_".join([f"{key}{value}" for key, value in params_to_update.items() if key != "KY"])
    directory  = os.path.join("TGLF/Simulations",simulation,param_dir)
    # List all files in the directory matching the pattern
    input_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith("ky_")]

    for file in input_files:
        print(f"Running tglf on {file}")
        # Run the TGLF command using subprocess
        subprocess.run(["tglf", "-e", file], check=True)

def read_tglf_files(template_path, params_to_update, ky_range):
    """

    Args:
        template_path (str): Path to the template TGLF file.
        params_to_update (dict): Dictionary of parameters to update with their new values.
        ky_range (list): List of KY values to scan.
    """

    output_base_dir = os.path.join("TGLF/Simulations",template_path)
    param_dir = "_".join([f"{key}{value}" for key, value in params_to_update.items() if key != "KY"])

    frequencies = [] 
    growth_rates = []

    for ky in ky_range:
        ky_dir = os.path.join(output_base_dir,param_dir, f"ky_{ky:.2f}","input.tglf")
        pyro = Pyro(gk_file=ky_dir,gk_code="TGLF")
        pyro.load_gk_output()
        data = pyro.gk_output.data
        growth_rates.append((list(data['growth_rate'].values[0])[0]))
        frequencies.append((list(data['mode_frequency'].values[0])[0]))

    return [frequencies, growth_rates]


def read_gs2_files(template_path, ky_range):
    

    output_base_dir = os.path.join("GS2/Simulations",template_path)

    frequencies = []
    growth_rates = []
    for ky in ky_range:
        ky_dir = os.path.join(output_base_dir, f"ky_{ky:.2f}","gs2.out.nc")
        try:
            # Open the NetCDF file
            dataset = xr.open_dataset(ky_dir)

            #print(dataset)
            growth_rates.append(dataset.isel(ky=0).isel(kx=0).isel(t=-1).isel(ri=1)["omega"].values)
            frequencies.append(dataset.isel(ky=0).isel(kx=0).isel(t=-1).isel(ri=0)["omega"].values)

        except:
            print(f"failed to read data from {ky_dir}")
            # Append dummy values if the file is not found or cannot be read
            growth_rates.append(0)
            frequencies.append(0)

    return [frequencies, growth_rates]


def run_TGLF(NBASIS_DIF, NBASIS_MIN, NXGRID, FILTER, WIDTH_DIF, WIDTH_MIN, THETA_TRAPPED):
    TGLF_data = []
    params_to_update = {
        "NBASIS_MAX": NBASIS_DIF+NBASIS_MIN,
        "NBASIS_MIN": NBASIS_MIN,
        "NXGRID": NXGRID,
        "FILTER": FILTER,
        "WIDTH": WIDTH_DIF+WIDTH_MIN,
        "WIDTH_MIN": WIDTH_MIN,
        "THETA_TRAPPED": THETA_TRAPPED,
    }

    for simulation in simulation_list:
        # Generate TGLF input files
        generate_tglf_files(simulation, params_to_update, KY_RANGE)

        run_tglf_on_files(simulation,params_to_update)

        TGLF_data.append(read_tglf_files(simulation,params_to_update, KY_RANGE))

    return TGLF_data

 

def Kernel_1(TGLF,GS2):
    """
    A Default Keneral For Comparing TGLF and GS2 Accuracy

    Parameters:
    TGLF (3D numpy array): Matrix of Mode Frequency, Growth Rate and **third** from TGLF for Each Simulation
    GS2 (3D numpy Array): Matrix of Mode Frequency, Growth Rate and **third** from GS2 for Each Simulation

    Returns:
    int: Naive Difference between TGLF and GS2
    """

    difference = np.sum(np.abs(np.array(TGLF)-np.array(GS2)))

    return difference


def Difference_Function(NBASIS_DIF,NBASIS_MIN,NXGRID,FILTER,WIDTH_DIF,WIDTH_MIN,THETA_TRAPPED):
    TGLF = run_TGLF(NBASIS_DIF,NBASIS_MIN,NXGRID,FILTER,WIDTH_DIF,WIDTH_MIN,THETA_TRAPPED)
    GS2 = []
    for simulation in simulation_list:
        GS2.append(read_gs2_files(simulation,KY_RANGE))
    params_to_update = {
        "NBASIS_MAX": NBASIS_DIF+NBASIS_MIN,
        "NBASIS_MIN": NBASIS_MIN,
        "NXGRID": NXGRID,
        "FILTER": FILTER,
        "WIDTH": WIDTH_DIF+WIDTH_MIN,
        "WIDTH_MIN": WIDTH_MIN,
        "THETA_TRAPPED": THETA_TRAPPED,
    }
    create_pdf_report(TGLF, GS2, params_to_update)
    return -1*Kernel_1(TGLF,GS2) # returns the negative so the optimizer minimizes its abosulte value


def plot_comparisons(TGLF_data, GS2_data):
    """
    Generate comparison plots for TGLF and GS2 data.

    Args:
        TGLF_data (list): TGLF simulation data.
        GS2_data (list): GS2 simulation data.

    Returns:
        list: List of matplotlib Figure objects.
    """
    plots = []
    for x in range(len(TGLF_data)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Frequency vs KY
        ax[0].plot(KY_RANGE, TGLF_data[x][0], label="Frequency TGLF", color="blue")
        ax[0].plot(KY_RANGE, GS2_data[x][0], label="Frequency GS2", color="red")
        ax[0].set_xlabel("KY")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("Frequency vs KY")
        ax[0].grid(True)
        ax[0].legend()

        # Growth Rate vs KY
        ax[1].plot(KY_RANGE, TGLF_data[x][1], label="Growth Rate TGLF", color="blue")
        ax[1].plot(KY_RANGE, GS2_data[x][1], label="Growth Rate GS2", color="red")
        ax[1].set_xlabel("KY")
        ax[1].set_ylabel("Growth Rate")
        ax[1].set_title("Growth Rate vs KY")
        ax[1].grid(True)
        ax[1].legend()

        # Growth Rate over KY vs KY
        ax[1].plot(KY_RANGE, TGLF_data[x][1]/KY_RANGE, label="Growth Rate/ky TGLF", color="blue")
        ax[1].plot(KY_RANGE, GS2_data[x][1]/KY_RANGE, label="Growth Rate/ky GS2", color="red")
        ax[1].set_xlabel("KY")
        ax[1].set_ylabel("Growth Rate/KY")
        ax[1].set_title("Growth Rate/KY vs KY")
        ax[1].grid(True)
        ax[1].legend()

        # Append the figure (not the module!)
        plots.append(fig)

    return plots

def create_pdf_report(TGLF_data, GS2_data, params):
    """
    Create a PDF report comparing TGLF and GS2 data.

    Args:
        TGLF_data (list): TGLF simulation data.
        GS2_data (list): GS2 simulation data.
        params (dict): Simulation parameters.
    """
    # Construct the output file path
    file_name = "_".join([f"{key}{value}" for key, value in params.items()]) + ".pdf"
    output_file = os.path.join("comparison", file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plots = plot_comparisons(TGLF_data, GS2_data)  # Assume this returns a list of matplotlib figures

    with PdfPages(output_file) as pdf:
        # First page: title and parameters
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
        ax.axis('off')

        title = "TGLF vs GS2 Comparison Report"
        ax.text(0.5, 0.95, title, fontsize=16, ha='center', va='top', weight='bold')

        ax.text(0.0, 0.9, "Simulation Parameters:", fontsize=12, weight='bold')
        y_pos = 0.88
        for key, value in params.items():
            ax.text(0.05, y_pos, f"- {key}: {value}", fontsize=11, ha='left')
            y_pos -= 0.03

        ax.text(0.0, y_pos, "Kenerl 1 Comparison:", fontsize=12, weight='bold')
        difference = Kernel_1(TGLF_data,GS2_data)
        ax.text(0.0, y_pos - 0.03, difference, fontsize=8)

        pdf.savefig(fig)
        plt.close(fig)

        # Add each plot as a new page
        for plot in plots:
            pdf.savefig(plot)
            plt.close(plot)

    print(f"PDF report saved as {output_file}")

def create_markdown_report(TGLF_data, GS2_data, params):
    """
    Create a Markdown report comparing TGLF and GS2 data.

    Args:
        TGLF_data (list): TGLF simulation data.
        GS2_data (list): GS2 simulation data.
        params (dict): Simulation parameters.
        output_file (str): Name of the output Markdown file.
    """
    # Construct the output file path
    file_name = "_".join([f"{key}{value}" for key, value in params.items()]) + ".md"
    output_file = os.path.join("comparison", file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plots = plot_comparisons(TGLF_data, GS2_data)   

    # Use the output_file in the rest of the code
    with open(output_file, "w") as f:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Title
        f.write(f"# TGLF vs GS2 Comparison Report\n\n")

        # Parameters
        f.write("## Simulation Parameters\n")
        for key, value in params.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")

        # Plots
        f.write("## Comparison Plots\n")
        for i, plot in enumerate(plots):
            # Save the plot to a BytesIO object
            buffer = BytesIO()
            plot.savefig(buffer, format="png")
            buffer.seek(0)

            # Encode the plot as a base64 string
            img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()

            # Embed the plot in the Markdown file
            f.write(f"### Plot {i + 1}\n")
            f.write(f"![Plot {i + 1}](data:image/png;base64,{img_base64})\n\n")

    print(f"Markdown report saved as {output_file}")



def run_optimation():
    # Bounded region of parameter space
    pbounds = {'NBASIS_DIF': (0, 8,int), 'NBASIS_MIN': (2, 10,int),'NXGRID':(10,100,int),'FILTER':(0,5),'WIDTH_dif':(0,1.9), 'WIDTH_MIN':(0.1,2),'THETA_TRAPPED':(0,1)}

    # Example: Run the Bayesian optimization process
    optimizer = BayesianOptimization(
        Difference_Function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )



if __name__ == "__main__":
    run_optimation()
    # Test run TGLF

    # TGLF_data = run_TGLF(2, 2, 10, 10, 0.2, 0.1, 0.5)
    # params = {
    #     "NBASIS_MAX": 2,
    #     "NBASIS_MIN": 2,
    #     "NXGRID": 10,
    #     "FILTER": 20,
    #     "WIDTH": 0.2,
    #     "WIDTH_MIN": 0.1,
    #     "THETA_TRAPPED": 0.5,
    # }
    # GS2_data = []
    # GS2_data.append(read_gs2_files(simulation_list[0], KY_RANGE))
    # print(Kernel_1(TGLF_data,GS2_data),params)
    # create_pdf_report(TGLF_data, GS2_data, params)
