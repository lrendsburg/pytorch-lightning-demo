import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x)) + x


class NeuralNetwork(nn.Module):
    def __init__(self, params):
        super().__init__()
        layer_widths = [1] + [params["layer_width"]] * params["num_layers"]
        self.layers = []
        for i, (in_features, out_features) in enumerate(
            zip(layer_widths, layer_widths[1:])
        ):
            setattr(self, f"linear{i+1}", nn.Linear(in_features, out_features))
            setattr(self, f"relu{i+1}", nn.ReLU())
            self.layers += [getattr(self, f"linear{i+1}"), getattr(self, f"relu{i+1}")]

        self.head = nn.Linear(params["layer_width"], 1)
        self.layers.append(self.head)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = NeuralNetwork({"num_layers": 3, "layer_width": 100})
    for name, param in model.named_parameters():
        print(name, param.shape)
