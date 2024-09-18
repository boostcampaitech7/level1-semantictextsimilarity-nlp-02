import torch.optim as optim

def get_optimizer(model, optimizer_config) :
    optimizer_type = optimizer_config['type']
    optimizer_params = optimizer_config['params']

    if optimizer_type == "Adam" :
        return optim.Adam(model.parameters(), **optimizer_params)
    else :
        raise ValueError(f"Unknown optimizer : {optimizer_type}")