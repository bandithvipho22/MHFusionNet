import torch
from torch import nn
from einops import rearrange
from fusionNet.model_poseformer import PoseTransformer
from common.arguments import parse_args
import time

args = parse_args()


# FC layer with Dense Connectivity
class DenseFC(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 linear_size=1024,
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
            # nn.BatchNorm2d(243),
            nn.LayerNorm([243, 17, self.linear_size]),
            # nn.PReLU(),
            # nn.Dropout(self.p_dropout),
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
    linear_size = 1024
    return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm([243, 17, linear_size]),
            # nn.BatchNorm2d(243),
            nn.PReLU(),
            nn.Dropout(p_dropout),
        )


class FusionNet(nn.Module):
    def __init__(self, num_frame, num_joints, hidden_chanel, out_chanel):
        super(FusionNet, self).__init__()

        self.hidden_chanel = hidden_chanel
        self.out_chanel = out_chanel

        self.feature_extractor = PoseTransformer(num_frame=num_frame, num_joints=num_joints, in_chans=3,
                                                 out_chans=hidden_chanel, embed_dim_ratio=32,
                                                 depth=4, num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
                                                 drop_path_rate=0.1)
        # DenseFC
        self.regression_head = DenseFC(
            in_channel=hidden_chanel * args.num_proposals,
            out_channel=out_chanel,
            linear_size=1024,  # 2048,
            num_stage=2,  # 2
            # p_dropout=0.01
        )

    def forward(self, x):
        # feature extractor (pose former)
        B, H, F, J, _ = x.shape
        out_fea_ext = []
        for i in range(args.num_proposals):
            output_extractor = self.feature_extractor(x[:, i])
            out_fea_ext.append(output_extractor)

        out_fea_ext = torch.cat(out_fea_ext, dim=3)  # → (B, F, J, H*C)

        # # Regression Head
        final_output = self.regression_head(out_fea_ext)

        return final_output


# if __name__ == '__main__':
#     frame = 243
#     c = 2
#     num_joint = 17
#     h = 5
#
#     encoder = FusionNet(num_frame=frame, num_joints=17, hidden_chanel=24, out_chanel=3).cuda()
#     # norm_1 = nn.LayerNorm(frame*5)
#
#     input = torch.randn(2, h, frame, 17, 3, dtype=torch.float32).cuda()
#     out_put = encoder(input)
#     print('Done!')
