import torch.nn as nn
import torch
from timm.models.layers import DropPath
from fusionNet.module.GCN_conv import ModulatedGraphConv
# from module.GCN_conv import ModulatedGraphConv


# GCN network with Residual Connectivity
class EmbeddingGCN(nn.Module):
    def __init__(self, adj, in_dim, out_dim, inter_dim, num_layer, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layer = num_layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adj = adj
        self.norm_gcn1 = norm_layer(in_dim)
        self.gcn1 = ModulatedGraphConv(in_dim, inter_dim, self.adj)
        self.gelu = nn.GELU()

        self.gcn_layers = []
        for i in range(num_layer):
            self.gcn_layers.append(gcn_block(adj, inter_dim, inter_dim, self.drop_path, norm_layer))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.gcn_final = ModulatedGraphConv(inter_dim, out_dim, self.adj)

    def forward(self, x_gcn):
        x_gcn = self.drop_path(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))

        for layer in self.gcn_layers:
            x_gcn = layer(x_gcn)

        x_gcn = self.gcn_final(x_gcn)
        return x_gcn


# # GCN Network with Dense Connectivity
# class DenseGCN(nn.Module):
#     def __init__(self, adj, in_dim, out_dim, inter_dim, num_layer, drop_path=0., norm_layer=nn.LayerNorm):
#         super(DenseGCN, self).__init__()
#         self.num_layer = num_layer
#         self.in_dim = in_dim
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.adj = adj
#         self.norm_gcn1 = norm_layer(17)
#         self.gcn1 = ModulatedGraphConv(in_dim, inter_dim, self.adj)
#         self.gelu = nn.GELU()
#
#         self.gcn_layers = []
#         for i in range(num_layer):
#             self.gcn_layers.append(gcn_block(adj, (inter_dim * (1 + i)), inter_dim, self.drop_path, norm_layer))
#         self.gcn_layers = nn.ModuleList(self.gcn_layers)
#
#         self.gcn_final = ModulatedGraphConv(inter_dim * (num_layer + 1), out_dim, self.adj)
#
#     def forward(self, x_gcn):
#         x_gcn = x_gcn.reshape(-1, 17, self.in_dim)
#         x_gcn = self.drop_path(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))
#
#         for layer in self.gcn_layers:
#             y = layer(x_gcn)
#             x_gcn = torch.cat((x_gcn, y), dim=2)
#
#         x_gcn = self.gcn_final(x_gcn)
#         return x_gcn.reshape(-1, 17 * 3)
#
#     def init_weights(m):
#         if isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
#             # nn.init.normal_(m.weight, std=1e-3)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0.0)
#         elif isinstance(m, nn.BatchNorm1d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#         else:
#             pass
#         return 0


# # GCN for local joint interaction
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
#         return x_gcn
#
#
def gcn_block(adj, input_size, output_size, p_dropout, norm_layer):
    return nn.Sequential(
        norm_layer(input_size),
        ModulatedGraphConv(input_size, output_size, adj),
        nn.GELU(),
        p_dropout,
    )
