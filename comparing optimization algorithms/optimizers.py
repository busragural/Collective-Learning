import torch.optim as optim

def get_optimizers():
    return {
        "SGD": lambda model: optim.SGD(model.parameters(), lr=0.01),
        "SGD with Momentum": lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "Adam": lambda model: optim.Adam(model.parameters(), lr=0.001)
    }
