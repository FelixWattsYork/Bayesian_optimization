import warnings
import numpy as np
import xarray as xr
import subprocess
import os
import matplotlib.pyplot as plt
import time
import sys
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from sklearn.gaussian_process.kernels import Matern
from matplotlib.backends.backend_pdf import PdfPages
from pyrokinetics import Pyro
import base64
from io import BytesIO

# suppress warnings about this being an experimental feature
warnings.filterwarnings(action="ignore")

simulation_list = ["r1","r2","r3","r4"]
KY_RANGE = np.arange(0.01, 1, 0.01)

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

def run_tglf_on_files_Compute(simulation, params_to_update):
    """
    Submit TGLF jobs to a Slurm cluster using sbatch and wait for completion.

    Args:
        simulation (str): Name of the simulation.
        params_to_update (dict): Dictionary of parameters to update with their new values.
    """
    param_dir = "_".join([f"{key}{value}" for key, value in params_to_update.items() if key != "KY"])
    directory = os.path.join("TGLF/Simulations", simulation, param_dir)

    # Create a directory for SLURM logs
    slurm_logs_dir = os.path.join(directory, "slurm_logs")
    os.makedirs(slurm_logs_dir, exist_ok=True)

    # List all input files in the directory matching the pattern
    input_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith("ky_")]

    job_ids = []  # List to store submitted job IDs

    for file in input_files:
        print(f"Submitting tglf job for {file}")

        # Create a Slurm job script
        job_script = f"""#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=nodes
#SBATCH --nodes=2  # Request 4 nodes
#SBATCH --ntasks-per-node=48  # Adjust based on the number of CPUs per node
#SBATCH --output={os.path.join(slurm_logs_dir, os.path.basename(file))}.out
#SBATCH --error={os.path.join(slurm_logs_dir,"e_", os.path.basename(file))}.err
srun --hint=nomultithread --distribution=block:block -n 96 tglf -e {file}
"""

        # Write the job script to a temporary file
        job_script_path = os.path.join(directory, f"job_{os.path.basename(file)}.job")

        # Ensure the file is overwritten if it already exists
        if os.path.exists(job_script_path):
            os.remove(job_script_path)

        with open(job_script_path, "w") as script_file:
            script_file.write(job_script)

        # Submit the job script using sbatch and capture the job ID
        result = subprocess.run(["sbatch", job_script_path], check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]  # Extract the job ID from sbatch output
        job_ids.append(job_id)

    return job_ids

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

    job_ids = []
    for simulation in simulation_list:
        # Generate TGLF input files
        generate_tglf_files(simulation, params_to_update, KY_RANGE)

        job_ids = job_ids + run_tglf_on_files_Compute(simulation,params_to_update)

    # Wait for all jobs to complete

    print("Waiting for all TGLF jobs to complete...")
    total_jobs = len(job_ids)

    while True:
        # Check the status of the jobs using squeue
        result = subprocess.run(["squeue", "--jobs", ",".join(job_ids), "--noheader"], capture_output=True, text=True)
        job_lines = result.stdout.strip().splitlines()  # Get the list of jobs still in the queue
        num_jobs_left = len(job_lines)  # Count the number of jobs left
        num_jobs_completed = total_jobs - num_jobs_left  # Calculate completed jobs

        # Display the loading bar
        progress = int((num_jobs_completed / total_jobs) * 50)  # Scale progress to 50 characters
        loading_bar = f"[{'#' * progress}{'.' * (50 - progress)}]"
        sys.stdout.write(f"\r{loading_bar} {num_jobs_completed}/{total_jobs} jobs completed")
        sys.stdout.flush()

        if num_jobs_left == 0:  # If no jobs are listed, they are all complete
            break

        time.sleep(10)  # Wait for 10 seconds before checking again

    print("\nAll TGLF jobs have completed.")

    for simulation in simulation_list:
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


def kernel_2(TGLF, GS2, ky_range=None, ky_weights=None):
    """
    Calculate the difference in growth rates between TGLF and GS2 for all simulations.

    Args:
        TGLF (list): 3D array containing TGLF data [frequency, growth_rate] for each simulation and KY value.
        GS2 (list): 3D array containing GS2 data [frequency, growth_rate] for each simulation and KY value.
        ky_range (list, optional): List of KY indices to consider. Defaults to all KY values.
        ky_weights (list, optional): List of weights for each KY value. Defaults to uniform weights.

    Returns:
        list: List of summed growth rate differences for each simulation.
    """
    if ky_range is None:
        ky_range = range(len(TGLF[0][1]))  # Default to all KY indices

    if ky_weights is None:
        ky_weights = [1] * len(ky_range)  # Default to uniform weights

    if len(ky_weights) != len(ky_range):
        raise ValueError("Length of ky_weights must match the length of ky_range.")

    differences = []

    for sim_idx, (tglf_sim, gs2_sim) in enumerate(zip(TGLF, GS2)):
        growth_rate_diff = 0
        for ky_idx, weight in zip(ky_range, ky_weights):
            tglf_growth_rate = tglf_sim[1][ky_idx]
            gs2_growth_rate = gs2_sim[1][ky_idx]
            growth_rate_diff += weight * abs(tglf_growth_rate - gs2_growth_rate)
        differences.append(growth_rate_diff)
    
    return np.sum(differences)





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
    return -1*kernel_2(TGLF,GS2) # returns the negative so the optimizer minimizes its abosulte value


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

def plot_comparison_for_simulation(TGLF_sim_data, GS2_sim_data, ky_range):
    """
    Generate a comparison plot for a single simulation's TGLF and GS2 data.

    Args:
        TGLF_sim_data (list): TGLF data [frequency, growth_rate] for a single simulation.
        GS2_sim_data (list): GS2 data [frequency, growth_rate] for a single simulation.
        ky_range (list): List of KY values.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object for the simulation.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Frequency vs KY
    ax[0].plot(ky_range, TGLF_sim_data[0], label="Frequency TGLF", color="blue")
    ax[0].plot(ky_range, GS2_sim_data[0], label="Frequency GS2", color="red")
    ax[0].set_xlabel("KY")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Frequency vs KY")
    ax[0].grid(True)
    ax[0].legend()

    # Growth Rate vs KY
    ax[1].plot(ky_range, TGLF_sim_data[1], label="Growth Rate TGLF", color="blue")
    ax[1].plot(ky_range, GS2_sim_data[1], label="Growth Rate GS2", color="red")
    ax[1].set_xlabel("KY")
    ax[1].set_ylabel("Growth Rate")
    ax[1].set_title("Growth Rate vs KY")
    ax[1].grid(True)
    ax[1].legend()

    return fig


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

    plots = []
    for sim_idx in range(len(TGLF_data)):
        plot = plot_comparison_for_simulation(TGLF_data[sim_idx], GS2_data[sim_idx], KY_RANGE)
        plots.append(plot)

    with PdfPages(output_file) as pdf:
        # First page: title and parameters
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
        ax.axis('off')

        title = "TGLF vs GS2 Comparison Report"
        ax.text(0.5, 0.9, title, fontsize=16, weight='bold', ha='center')

        ax.text(0.0, 0.8, "Simulation Parameters:", fontsize=12, weight='bold')
        y_pos = 0.75
        for key, value in params.items():
            ax.text(0.0, y_pos, f"{key}: {value}", fontsize=11, ha='left')
            y_pos -= 0.03

        ax.text(0.0, y_pos, "Kernel 2 Comparison:", fontsize=12, weight='bold')
        difference = kernel_2(TGLF_data, GS2_data)
        ax.text(0.0, y_pos - 0.03, f"Difference: {difference}", fontsize=11)

        pdf.savefig(fig)
        plt.close(fig)


        # Add each plot as a new page
        for plot in plots:
            # Add a heading for the simulation
            fig, ax = plt.subplots(figsize=(8.27, 1))  # Small figure for the heading
            ax.axis('off')
            simulation_name = simulation_list[sim_idx]
            ax.text(0.5, 0.5, f"Simulation: {simulation_name}", fontsize=14, weight='bold', ha='center')
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

def plot_bayesian_optimization_results(optimizer, output_file="Bayesian_Optimization_Results.pdf"):
    """
    Plot the results of the Bayesian optimization process and save them to a PDF.

    Args:
        optimizer (BayesianOptimization): The BayesianOptimization object after running the optimization.
        output_file (str): Name of the output PDF file.
    """
    # Extract the results
    iterations = list(range(1, len(optimizer.res) + 1))
    target_values = [res["target"] for res in optimizer.res]

    # Create a PDF file
    with PdfPages(output_file) as pdf:
        # Plot the target values over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, target_values, marker="o", linestyle="-", color="b", label="Target Value")
        plt.xlabel("Iteration")
        plt.ylabel("Target Value")
        plt.title("Bayesian Optimization Results")
        plt.grid(True)
        plt.legend()
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()

        # Add a summary page with the best result
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
        ax.axis('off')

        title = "Bayesian Optimization Summary"
        ax.text(0.5, 0.9, title, fontsize=16, weight='bold', ha='center')

        best_result = optimizer.max
        ax.text(0.0, 0.8, "Best Result:", fontsize=12, weight='bold')
        ax.text(0.0, 0.75, f"Target: {best_result['target']}", fontsize=11)
        ax.text(0.0, 0.7, "Parameters:", fontsize=12, weight='bold')
        y_pos = 0.65
        for key, value in best_result["params"].items():
            ax.text(0.0, y_pos, f"{key}: {value}", fontsize=11)
            y_pos -= 0.03

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF report saved as {output_file}")

def run_optimation():
    # Bounded region of parameter space
    pbounds = {'NBASIS_DIF': (0, 8,int), 'NBASIS_MIN': (2, 10,int),'NXGRID':(10,100,int),'FILTER':(0,5),'WIDTH_DIF':(0,1.9), 'WIDTH_MIN':(0.1,2),'THETA_TRAPPED':(0,1)}

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

    # Generate a PDF report with the results
    plot_bayesian_optimization_results(optimizer, output_file="Bayesian_Optimization_Results.pdf")



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
