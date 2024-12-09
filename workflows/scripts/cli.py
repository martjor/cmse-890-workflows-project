import click
import main
import os
from networkx import to_scipy_sparse_array
from scipy.sparse import save_npz
from math import ceil

@click.command()
@click.argument('output',type=click.Path())
@click.argument('config-file',type=click.Path(exists=True))
def sample(output,
         config_file):
    # Load script configuration
    config = {}
    if config_file:
        config = main.io.load_config(config_file)

    # Get function arguments
    arguments = main.io.get_arguments(main.sample)
    
    # Verify arguments
    for argument in arguments:
        if config.get(argument,None) is None:
            message = f"Input argument '{argument}' missing from parameter file"
            raise ValueError(message)
        
    # Evaluate function
    data, metadata = main.sample(**config)
    
    # Create output directory
    main.io.mkdir(output)
    
    # Write data
    paths = [""] * metadata.shape[0]
    for i, graph in enumerate(data):
        paths[i] = os.path.join(output,f"graph_{i:02d}.npz")
        save_npz(paths[i],to_scipy_sparse_array(graph))
    
    # Write metadata
    metadata['graph_file'] = paths
    metadata.set_index('graph_file',inplace=True)
    metadata.to_csv(os.path.join(output,'metadata.csv'))

@click.command()
@click.argument('output',type=click.Path())
def characterize(output):
    # Instantiate a database
    db = main.Database()
    table_graphs = db.graphs
    
    # Batch-process graphs
    batch_size = 50
    n_graphs = table_graphs.shape[0]
    n_batches = int(ceil(n_graphs / batch_size))
    tags = [0] * n_graphs
    files = [0] * n_graphs
    data = [0] * n_batches
    
    for i in range(n_batches):
        idx_low = i * batch_size
        idx_high = min(idx_low + batch_size, n_graphs)
        
        # Load graphs at the beginning of each batch
        graphs = [0] * (idx_high - idx_low)
        for j, idx_global in enumerate(range(idx_low,idx_high)):
            tags[idx_global] = f"_{idx_global:04d}"
            graphs[j] = main.io.load_graph(table_graphs['graph_file'][idx_global])
                
        # Process graphs
        char = main.Characterization(graphs)
        dist_degree, dist_paths, data[i] = char.report()
        
        # Write distributions
        main.io.mkdir(output)
        for j, idx_global in enumerate(range(idx_low,idx_high)):
            # Create file names
            tag = tags[idx_global]
            path = os.path.join(output,'dist')
            files[idx_global] = [path + f"_deg{tag}.npy", path + f"_dist{tag}.npy"]
            
            # Save files
            np.save(files[idx_global][0],dist_degree[j])
            np.save(files[idx_global][1],dist_paths[j])
            
        # Construct global dataframe
        metadata = pd.Concat(data,columns=columns,ignore_index=True)
        metadata['Graph'] = table_graphs['file_name']
        metadata['files'] = files
        metadata.set_index('Graph')
        
        # Save dataframe
        metadata.to_csv(os.path.join(output,'metadata.csv'))
        
        
            
            
            
        
        
                
        
    