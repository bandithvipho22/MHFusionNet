import torch
from torch import nn
from einops import rearrange
from fusionNet.model_poseformer import PoseTransformer
from common.arguments_edited import parse_args
from fusionNet.module.enc_transformer_v2 import Transformer as TransformerEncoder
from fusionNet.module.dec_transformer import Transformer as TransformerDecoder
import time

args = parse_args()


class FusionNet(nn.Module):
    def __init__(self, num_frame, num_joints, hidden_chanel, out_chanel):
        super().__init__()

        self.feature_extractor = PoseTransformer(num_frame=num_frame, num_joints=num_joints, in_chans=3,
                                                 out_chans=hidden_chanel, embed_dim_ratio=32,
                                                 depth=4, num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
                                                 drop_path_rate=0.1)

        # # # Transformer encoder
        # self.transformer_encoder = TransformerEncoder(depth=4, embed_dim=num_frame * args.num_proposals,
        #                                               mlp_hidden_dim=num_frame * 2, h=9,
        #                                               length=num_joints * hidden_chanel)

        # Transformer decoder
        self.transformer_decoder = TransformerDecoder(depth=4, embed_dim=num_frame,
                                                      mlp_hidden_dim=num_frame * 2, h=9,
                                                      length=num_joints * hidden_chanel)

        # # Regression Head
        # self.regression_head = nn.Sequential(
        #     # nn.Conv2d(in_channels=num_frame*args.num_proposals, out_channels=num_frame, kernel_size=1),
        #     # nn.Linear(in_features=hidden_chanel, out_features=32),
        #     nn.Linear(in_features=hidden_chanel*args.num_proposals, out_features=32),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=out_chanel)
        # )

        self.regression_head = nn.Sequential(
            # nn.LayerNorm(hidden_chanel * args.num_proposals),
            nn.Linear(in_features=hidden_chanel * args.num_proposals, out_features=64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(in_features=128, out_features=64),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=out_chanel)
        )

        # self.regression_head = nn.Sequential(
        #     # nn.LayerNorm(hidden_chanel * args.num_proposals),
        #     nn.Linear(in_features=hidden_chanel * args.num_proposals, out_features=68),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #
        #     nn.Linear(in_features=68, out_features=42),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #
        #     # nn.Linear(in_features=42, out_features=64),
        #     # nn.ReLU(),
        #
        #     nn.Linear(in_features=42, out_features=out_chanel)
        # )

        self.norm_decoder = nn.LayerNorm(num_frame)
        self.pose_embed = nn.Parameter(torch.zeros(1, num_frame, num_joints, hidden_chanel))

    def forward(self, x):
        # feature extractor (pose former)
        out_fea_ext = []
        for i in range(args.num_proposals):
            output_extractor = self.feature_extractor(x[:, i])
            out_fea_ext.append(output_extractor)

        out_fea_ext = torch.stack(out_fea_ext, dim=1)

        # Reshape output feature from Feature Extractor
        B, H, F, J, C = out_fea_ext.shape
        out_fea_ext = out_fea_ext.reshape(B, H, F, J, C)

        # Transformer decoder
        out_decoders = []
        for i in range(args.num_proposals):
            f_dec = out_fea_ext[:, i] + self.pose_embed
            f_dec = rearrange(f_dec, 'b f j c -> b (j c) f').contiguous()
            out_decoder = self.transformer_decoder(f_dec)
            out_decoder = rearrange(out_decoder, 'b (j c) f -> b f j c', j=J).contiguous()
            out_decoders.append(out_decoder)  # (b, h, f, j, c)

        # Regression Head
        out_decoders = torch.cat(out_decoders, dim=3)
        final_output = self.regression_head(out_decoders)

        return final_output


if __name__ == '__main__':
    frame = 243
    c = 2
    num_joint = 17

    encoder = FusionNet(num_frame=243, num_joints=17, hidden_chanel=24, out_chanel=3).cuda()
    norm_1 = nn.LayerNorm(frame*5)

    input = torch.randn(10, 5, 243, 17, 3, dtype=torch.float32).cuda()
    out_put = encoder(input)
    print('Done!')


