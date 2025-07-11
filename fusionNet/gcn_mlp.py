import torch.nn as nn
import torch
from timm.models.layers import DropPath
from fusionNet.module.GCN_conv import ModulatedGraphConv, adj_mx_from_skeleton
# from module.GCN_conv import ModulatedGraphConv
from fusionNet.model_poseformer import PoseTransformer
from common.arguments import parse_args
# import time

args = parse_args()


# GCN network with Residual Connectivity
# class EmbeddingGCN(nn.Module):
#     def __init__(self, adj, in_dim, out_dim, inter_dim, num_layer, drop_path=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.num_layer = num_layer
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.adj = adj
#         self.norm_gcn1 = norm_layer(in_dim)
#         self.gcn1 = ModulatedGraphConv(in_dim, inter_dim, self.adj)
#         self.gelu = nn.GELU()
#
#         self.gcn_layers = []
#         for i in range(num_layer):
#             self.gcn_layers.append(gcn_block(adj, inter_dim, inter_dim, self.drop_path, norm_layer))
#         self.gcn_layers = nn.ModuleList(self.gcn_layers)
#
#         self.gcn_final = ModulatedGraphConv(inter_dim, out_dim, self.adj)
#
#     def forward(self, x_gcn):
#         x_gcn = self.drop_path(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))
#
#         for layer in self.gcn_layers:
#             x_gcn = layer(x_gcn)
#
#         x_gcn = self.gcn_final(x_gcn)
#         return x_gcn
#
#
# def gcn_block(adj, input_size, output_size, p_dropout, norm_layer):
#     return nn.Sequential(
#         norm_layer(input_size),
#         ModulatedGraphConv(input_size, output_size, adj),
#         nn.GELU(),
#         p_dropout,
#     )

# GCN for local joint interaction
# class LJC(nn.Module):
#     def __init__(self, adj, in_dim, out_dim, inter_dim, drop_path=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.adj = adj
#         self.norm_gcn1 = norm_layer(in_dim)
#         self.gcn1 = ModulatedGraphConv(in_dim, inter_dim, self.adj)
#         self.gelu = nn.GELU()
#         self.gcn2 = ModulatedGraphConv(inter_dim, out_dim, self.adj)
#         self.norm_gcn2 = norm_layer(out_dim)
#
#     def forward(self, x_gcn):
#         x_gcn = x_gcn + self.drop_path(self.norm_gcn2(self.gcn2(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))))
#         out = self.gcn1(self.norm_gcn1(x_gcn))
#         out = self.gelu(out)
#         out = self.gcn2(out)
#         out = self.norm_gcn2(out)
#         return x_gcn

# GCN Network with Dense Connectivity
class DenseGCN(nn.Module):
    def __init__(self, adj, in_dim, out_dim, inter_dim, num_layer, drop_path=0., norm_layer=nn.LayerNorm):
        super(DenseGCN, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adj = adj
        self.norm_gcn1 = norm_layer(3)  # default 17
        self.gcn1 = ModulatedGraphConv(in_dim, inter_dim, self.adj)
        self.gelu = nn.GELU()

        self.gcn_layers = []
        for i in range(num_layer):
            self.gcn_layers.append(gcn_block(adj, (inter_dim * (1 + i)), inter_dim, self.drop_path, norm_layer))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.gcn_final = ModulatedGraphConv(inter_dim * (num_layer + 1), out_dim, self.adj)

    def forward(self, x_gcn):
        x_gcn = x_gcn.reshape(-1, 17, self.in_dim)
        x_gcn = self.drop_path(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))

        for layer in self.gcn_layers:
            y = layer(x_gcn)
            x_gcn = torch.cat((x_gcn, y), dim=2)

        x_gcn = self.gcn_final(x_gcn)
        return x_gcn.reshape(-1, 17 * 3)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
            # nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            pass
        return 0


def gcn_block(adj, input_size, output_size, p_dropout, norm_layer):
    return nn.Sequential(
        norm_layer(input_size),
        ModulatedGraphConv(input_size, output_size, adj),
        nn.GELU(),
        p_dropout,
    )


class FusionNet(nn.Module):
    def __init__(self, num_frame, num_joints, hidden_chanel, out_chanel):
        super(FusionNet, self).__init__()

        self.hidden_chanel = hidden_chanel
        self.out_chanel = out_chanel

        self.adj_mx_from_skeleton = adj_mx_from_skeleton()

        # GCN wirh Dense connectivity feature extractor
        self.GCN_extractor = DenseGCN(
            adj=self.adj_mx_from_skeleton,
            # adj=[num_joints*num_joints],
            in_dim=3,
            out_dim=self.hidden_chanel,
            # out_dim=3 * args.num_proposals,
            inter_dim=64,  # can test: 128, 1024
            num_layer=8,
            drop_path=0.01,
            # norm_layer=nn.LayerNorm
        )

        self.regression_head = nn.Sequential(
            # nn.LayerNorm(hidden_chanel * args.num_proposals),
            nn.Linear(in_features=hidden_chanel * args.num_proposals, out_features=68),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(in_features=68, out_features=42),
            nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(in_features=42, out_features=64),
            # nn.ReLU(),

            nn.Linear(in_features=42, out_features=out_chanel)
        )

    def forward(self, x):
        B, H, F, J, _ = x.shape
        out_fea_ext = []
        x = x[:, :, 0, :, :]
        for i in range(args.num_proposals):
            output_extractor = self.GCN_extractor(x[:, i])
            out_fea_ext.append(output_extractor)

        out_fea_ext = torch.cat(out_fea_ext, dim=1)
        out_fea_ext = out_fea_ext.reshape(B, F, J, self.hidden_chanel*args.num_proposals)

        # fused_input = fused_input.permute(0, 1, 3, 2)
        # fused_input = fused_input.reshape(B, J, -1)
        # fused_input = out_fea_ext[:, 0, :, :]
        # fused_input = fused_input.permute(0, 2, 1)

        final_output = self.regression_head(out_fea_ext)
        # final_output = final_output.reshape(B, F, J, self.out_chanel)

        return final_output


# if __name__ == '__main__':
#     frame = 1
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
