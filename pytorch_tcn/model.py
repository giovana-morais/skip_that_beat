import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dilation_rate, n_filters, kernel_size, padding, dropout_rate):
        """
        parameters:
            dilation_rate : int
                dilation_rate
            n_filters : int
                number of filters
            kernel_size : int or tuple
                conv kernel size
            padding : str
                padding (["valid", "same"])
            dropout_rate : float
                dropout rate
        """
        super().__init__()
        in_channels = 20

        self.res = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            padding=padding,
        )
        self.conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=padding,
        )
        self.conv_2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation_rate * 2,
            padding=padding,
        )
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv_3 = nn.Conv1d(
            in_channels=2 * n_filters,
            out_channels=n_filters,
            kernel_size=1,
            padding=padding,
        )
        return

    def forward(self, x):
        res_x = self.res(x)
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(x)
        concat = torch.cat([conv_1, conv_2], dim=1)
        out = self.elu(concat)
        out = self.dropout(out)
        out = self.conv_3(out)

        return res_x + out, out


class TCN(nn.Module):
    def __init__(
        self,
        n_filters=20,
        kernel_size=5,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        padding="same",
        dropout_rate=0.15,
    ):
        super().__init__()

        self.tcn_layers = nn.ModuleDict({})
        for idx, d in enumerate(dilations):
            self.tcn_layers[f"tcn_{idx}"] = ResBlock(
                d, n_filters, kernel_size, padding, dropout_rate
            )

        self.activation = nn.ELU()
        return

    def forward(self, x):
        skip_connections = []
        for tcn_i in self.tcn_layers:
            # feed the output of the previous layer into the next layer
            # increase dilation rate for each consecutive layer
            x, skip_out = self.tcn_layers[tcn_i](x)
            # print("tcn x.shape", x.shape)
            # print("tcn skip_out", skip_out.shape)
            skip_connections.append(skip_out)

        x = self.activation(x)

        skip = torch.stack(skip_connections, dim=-1).sum(dim=-1)

        return x, skip


class MultiTracker(nn.Module):
    def __init__(
        self, n_filters, n_dilations, kernel_size, dropout_rate, verbose=False
    ):
        super().__init__()
        padding = "same"
        self.verbose = verbose
        self.dropout_rate = dropout_rate

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(3, 3),
            stride=1,
            padding="valid",
        )
        self.elu_1 = nn.ELU()
        self.mp_1 = nn.MaxPool2d((1, 3))
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.conv_2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=(1, 10),
            padding="valid",
        )
        self.elu_2 = nn.ELU()
        self.mp_2 = nn.MaxPool2d((1, 3))
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.conv_3 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=(3, 3),
            padding="valid",
        )
        self.elu_3 = nn.ELU()
        self.mp_3 = nn.MaxPool2d((1, 3))
        self.dropout_3 = nn.Dropout(dropout_rate)

        dilations = [2**i for i in range(n_dilations)]

        self.tcn = TCN(n_filters, kernel_size, dilations, padding, dropout_rate)

        # beat head
        self.beats_dropout = nn.Dropout(dropout_rate)
        self.beats_dense = nn.Linear(n_filters, 1)
        self.beats_act = nn.Sigmoid()

        # downbeat head
        self.downbeats_dropout = nn.Dropout(dropout_rate)
        self.downbeats_dense = nn.Linear(n_filters, 1)
        self.downbeats_act = nn.Sigmoid()

        # tempo head
        self.tempo_dropout = nn.Dropout(dropout_rate)
        self.tempo_dense = nn.Linear(n_filters, 300)
        self.tempo_act = nn.Softmax(dim=1)

    def forward(self, x):
        if self.verbose:
            print("input shape:", x.shape)
        x = self.conv_1(x)
        x = self.elu_1(x)
        x = self.mp_1(x)
        x = self.dropout_1(x)
        if self.verbose:
            print("block1 out", x.shape)

        x = self.conv_2(x)
        x = self.elu_2(x)
        x = self.mp_2(x)
        x = self.dropout_2(x)
        if self.verbose:
            print("block2 out", x.shape)

        x = self.conv_3(x)
        x = self.elu_3(x)
        x = self.mp_3(x)
        x = self.dropout_3(x)
        if self.verbose:
            print("block3 out", x.shape)

        x = torch.squeeze(x, -1)
        if self.verbose:
            print("reshape", x.shape)

        x, skip = self.tcn(x)
        if self.verbose:
            print("tcn x out", x.shape)
            print("skip out", skip.shape)

        # reshape x so it fits our linear layer
        x = x.transpose(-2, -1)
        if self.verbose:
            print("tcn tranposed", x.shape)

        # beats head
        beats = self.beats_dropout(x)
        beats = self.beats_dense(beats)
        beats = self.beats_act(beats)

        if self.verbose:
            print("beats", beats.shape)

        # downbeats head
        downbeats = self.downbeats_dropout(x)
        downbeats = self.downbeats_dense(downbeats)
        downbeats = self.downbeats_act(downbeats)
        if self.verbose:
            print("downbeats", downbeats.shape)

        # tempo head
        # tempo = self.tempo_dropout(skip)
        # print("tempo before pooling", tempo.shape)
        # # global avg pooling over the time dimension
        # tempo = tempo.mean(dim=-1)
        # print("tempo after pooling", tempo.shape)
        # # add gaussian noise
        # gaussian_noise = torch.zeros_like(tempo).normal_(0, self.dropout_rate)
        # tempo = tempo + gaussian_noise
        # tempo = self.tempo_dense(tempo)
        # tempo = self.tempo_act(tempo)
        # print("tempo", tempo.shape)

        activations = {}
        activations["beats"] = beats
        activations["downbeats"] = downbeats
        # activations["tempo"] = tempo

        return activations


if __name__ == "__main__":
    test_input = torch.randn([1, 1, 3006, 81])
    n_filters = 20
    kernel_size = 5
    padding = "same"
    dropout = 0.15

    # beats , downbeats, tempo = MultiTracker(n_filters, n_dilations=11, kernel_size=kernel_size, dropout_rate=dropout)(test_input)
    beats, downbeats = MultiTracker(
        n_filters, n_dilations=11, kernel_size=kernel_size, dropout_rate=dropout
    )(test_input)

    print("\noutputs")
    print("beats", beats.shape)
    # print("downbeats", downbeats.shape)
    # print("tempo", tempo.shape)
