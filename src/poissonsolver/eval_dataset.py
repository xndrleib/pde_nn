import argparse
from pathlib import Path

import pandas as pd
import yaml

# From PlasmaNet
from src.poissonsolver.network import PoissonNetwork

latex_dict = {
    'Eresidual': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_1$',
    'Einf_norm': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_\infty$',
    'residual': r'$||u_\mathrm{out} - u_\mathrm{target}||_1$',
    'inf_norm': r'$||u_\mathrm{out} - u_\mathrm{target}||_1$',
    'phi11': r'$||u^{out}_\mathrm{11} - u^{target}_\mathrm{11}||_1$',
    'phi12': r'$||u^{out}_\mathrm{12} - u^{targte}_\mathrm{12}||_1$',
    'phi21': r'$||u^{out}_\mathrm{21} - u^{target}_\mathrm{21}||_1$',
    'phi22': r'$||u^{out}_\mathrm{22} - u^{target}_\mathrm{22}||_1$',
}


def main():
    args = argparse.ArgumentParser(description='PoissonNetwork evaluation')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-fn', '--filename', default='metrics', type=str,
                      help='Name of the h5 file of the DataFrame in the casename directory')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Neural network configuration
    network_cfg = config['network']

    # datadir
    data_dir = Path(network_cfg['casename'])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Dataset to evaluate
    dataset_name = list(config['datasets'].keys())[0]
    dataset_loc = Path(config['datasets'][dataset_name])

    # Metrics and number of metrics
    metrics = network_cfg['metrics']

    # Global dataframe
    df_columns = ['nn_name', 'nn_type', 'rf_global', 'nbranches', 'depth', 'ks', 'ds_name', 'ds_type',
                  'test_res', 'train_res', 'metric_name', 'value']
    df = pd.DataFrame(columns=df_columns)

    config['network']['eval'] = config['eval']
    config['network']['resume'] = network_cfg['resume']
    config['network']['arch'] = network_cfg['arch']

    # Initialize the PoissonNetwork with the provided config
    poisson_nn = PoissonNetwork(config['network'])

    # Extract network global properties
    rf_global = poisson_nn.model.rf_global_x  # or rg_global_y
    nbranches = poisson_nn.model.n_scales
    depths = poisson_nn.model.depths
    ks = poisson_nn.model.kernel_sizes[0]
    test_res = config['eval']['nnx']
    train_res = network_cfg['globals']['nnx']

    # Evaluate on the specified dataset
    metrics_tmp = poisson_nn.evaluate(dataset_loc, data_dir / dataset_name, plot=True, save_data=True)

    # Save metrics to dataframe
    rows = []
    for metric_name in metrics:
        tmp_dict = {
            'nn_name': network_cfg['casename'].split('/')[1],
            'nn_type': type(poisson_nn.model).__name__,
            'rf_global': rf_global,
            'nbranches': nbranches,
            'depth': sum(depths),
            'ks': ks,
            'ds_name': dataset_name,
            'ds_type': dataset_name.split('_')[0],  # Assuming type is indicated by the first part of the name
            'test_res': test_res,
            'train_res': train_res,
            'metric_name': metric_name,
            'value': metrics_tmp._data.average[metric_name]
        }
        rows.append(tmp_dict)

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # Save the dataframe
    h5filename = args.filename + '.h5'
    df.to_hdf(data_dir / h5filename, key='df', mode='w')


if __name__ == '__main__':
    main()
