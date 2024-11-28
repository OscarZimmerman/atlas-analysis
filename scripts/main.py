import uproot
import numpy as np
import awkward as ak
from scripts.data_processing import process_tree
from scripts.data_analysis import fit_data, plot_results
from scripts.utils import load_config

def main():
    config = load_config('config/config.yaml')
    
    variables = config['variables']
    path = config['data_path']
    samples = config['samples']
    
    all_data = []
    
    for sample in samples:
        file_path = f"{path}/{sample}.root"
        print(f"Processing {file_path}...")
        
        with uproot.open(f"{file_path}:analysis") as tree:
            data = process_tree(tree, variables, fraction=config['fraction'])
            all_data.append(data)
    
    combined_data = ak.concatenate(all_data)
    
    bin_edges = np.linspace(config['xmin'], config['xmax'], config['num_bins'] + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    data_x, _ = np.histogram(ak.to_numpy(combined_data['mass']), bins=bin_edges)
    data_x_errors = np.sqrt(data_x)
    
    fit_result = fit_data(bin_centres, data_x, data_x_errors)
    
    background = np.polyval([fit_result.params[f'c{i}'].value for i in range(5)], bin_centres)
    
    output_dir = "data"

    plot_results(bin_centres, data_x, data_x_errors, fit_result, background,
                 config['step_size'], config['xmin'], config['xmax'], output_dir)

if __name__ == "__main__":
    main()
