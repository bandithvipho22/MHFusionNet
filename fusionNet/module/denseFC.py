import torch.nn as nn
import torch
# from timm.models.layers import DropPath
# from Modulated_GCN.GCN_conv import ModulatedGraphConv


# FC layer with Dense Connectivity
class DenseFC(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 linear_size=2048,
                 num_stage=3,
                 p_dropout=0.5,
                 ):
        super(DenseFC, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.input_size = in_channel
        self.output_size = out_channel

        self.dense_in = nn.Sequential(
            nn.Linear(self.input_size, self.linear_size),
            nn.BatchNorm2d(243),
            nn.PReLU(),
            nn.Dropout(self.p_dropout),
        )

        self.linear_stages = []
        for i in range(num_stage):
            self.linear_stages.append(linear_block(linear_size * (1 + i), linear_size, p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.dense_out = nn.Sequential(
            nn.Linear(self.linear_size * (self.num_stage + 1), self.output_size),
        )

    def forward(self, input):

        x = self.dense_in(input)
        for blk in self.linear_stages:
            y = blk(x)
            # Concatenate the input and output of each block on the channel dimension
            x = torch.cat((x, y), dim=3)
        output = self.dense_out(x)

        return output


def linear_block(input_size, output_size, p_dropout):
    return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm2d(243),
            nn.PReLU(),
            nn.Dropout(p_dropout),
        )


# if __name__ == '__main__':
#     frame = 243
#     c = 24
#     num_joint = 17
#
#     encoder = DenseFC(in_channel=5*c, out_channel=3, linear_size=2048, num_stage=2).cuda()
#     # norm_1 = nn.LayerNorm(frame*5)
#
#     input = torch.randn(2, 243, 17, 5*c, dtype=torch.float32).cuda()
#     out_put = encoder(input)
#     print('Done!')
