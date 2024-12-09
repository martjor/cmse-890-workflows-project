import pandas as 
import yaml

# Load DataFrame
    parameters = pd.read_csv(input[0])

    # Convert parameters to dictionary
    idx = int(wildcards.target_id)
    parameters = parameters.iloc[idx].to_dict()

    # Write parameters
    with open(output[0],'w') as file:
        yaml.dump(parameters,file,default_flow_style=False)