import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPFactory:
    def __init__(self, hidden_size=688, num_layers=4):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def create(self, input_size, num_classes):
        return SimpleMLP(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes
        )
        
        '''
        from cl_framework.simple_mlp import MLPFactory
        mlp_factory = MLPFactory(hidden_size=688, num_layers=4)
        model = mlp_factory.create(input_size=input_size, num_classes=n_labels)
        '''