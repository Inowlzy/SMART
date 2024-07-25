import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

def relative_to_absolute(q, rq):
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    x = final_x[:, :, :l, (l - 1):].permute(0, 2, 3, 1).squeeze(-1)
    rq = rq.squeeze(1)
    rqt = rq.transpose(1, 2)
    r = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(rqt, rq)), rqt), x)
    r = r.transpose(1, 2)

    return r

def rel_pos_emb_1d(q, rel_emb, shared_heads):
   if shared_heads:
       emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
   else:
       emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
   return relative_to_absolute(emb, rq=q)

class RelPosEmb1DAISummer(nn.Module):
   def __init__(self, tokens, dim_head, heads=None):
       super().__init__()
       scale = dim_head ** -0.5
       self.shared_heads = heads if heads is not None else True
       if self.shared_heads:
           self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale).to('cuda:0')
       else:
           self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale).to('cuda:0')
   def forward(self, q):
       return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)

class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SceneMaskAttention(nn.Module):
    def __init__(self):
        super(SceneMaskAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 128 * 1, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, scene_feature):
        output = self.relu(self.conv1(input_features)).squeeze(-1)
        output = self.pool(output).unsqueeze(-1)
        output = self.relu(self.conv2(output)).squeeze(-1)
        output = self.pool(output).unsqueeze(-1)
        output = output.view(output.size(0), -1)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        output = self.sigmoid(output)
        binary_output = output
        binary_output = binary_output.view(scene_feature.size(0), 1, 1)
        scene_feature = scene_feature * binary_output
        return scene_feature, binary_output

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128*255, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward_once(self, x):
        x = x.reshape(-1, 64*256, 2).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=8)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.abs(output1 - output2)
        output = self.fc2(output)
        output = F.sigmoid(output)
        return output

class ScaleNetwork(nn.Module):
    def __init__(self):
        super(ScaleNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128*14, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward_once(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.abs(output1 - output2)
        output = self.fc2(output)
        output = F.sigmoid(output)
        return output

class ObjectSeparatedAttention(nn.Module):
    def __init__(self, seq_len, num_heads):
        super(ObjectSeparatedAttention, self).__init__()

        self.self_attention_person = nn.MultiheadAttention(3, num_heads)
        self.depth_association_network = ScaleNetwork()
        self.mask_association_network = SiameseNetwork()

    def forward(self, scene_sequence1, scene_sequence2):
        # person spatial motion feature
        input_sequence1 = scene_sequence1[:, 0, :, :]
        input_sequence2 = scene_sequence2[:, 0, :, :, :]
        input_sequence2 = torch.mean(input_sequence2, dim=2, keepdim=False)
        input_sequence = torch.cat((input_sequence1, input_sequence2), dim=2)
        input_sequence_reposembed = input_sequence + RelPosEmb1DAISummer(tokens=64, dim_head=3)(input_sequence.unsqueeze(1))
        del input_sequence1, input_sequence2
        spatial_motion_feature = self.self_attention_person(input_sequence, input_sequence_reposembed, input_sequence)[0]
        del input_sequence, input_sequence_reposembed

        # scene related feature
        depth_related_feature = self.depth_association_network(scene_sequence1[:,1,:,:].permute(0,2,1), scene_sequence1[:,0,:,:].permute(0,2,1))
        del scene_sequence1
        mask_related_feature = self.mask_association_network(scene_sequence2[:,1,:,:,:], scene_sequence2[:,0,:,:,:])
        del scene_sequence2
        scene_feature = torch.cat((depth_related_feature, mask_related_feature), dim=1)
        del depth_related_feature, mask_related_feature

        return spatial_motion_feature, scene_feature

class FeatureFusionNetwork(nn.Module):
    def __init__(self):
        super(FeatureFusionNetwork, self).__init__()
        self.channel_attention1 = ECALayer(channel=2)
        self.channel_attention2 = ECALayer(channel=2)
        self.scene_attention = SceneMaskAttention()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(192, 512)
        self.fc2 = nn.Linear(2, 512)
        self.fc3 = nn.Linear(1536, 512)

    def forward(self, action_tensor, spatial_motion_feature, scene_feature):
        spatial_motion_feature = self.relu1(self.fc1(spatial_motion_feature.reshape(-1, 192))).unsqueeze(1)
        scene_feature = self.relu2(self.fc2(scene_feature)).unsqueeze(1)
        fusion_feature = torch.cat((action_tensor, spatial_motion_feature), dim=1).unsqueeze(-1)
        fusion_feature = self.channel_attention1(fusion_feature)
        scene_feature, scene_weight = self.scene_attention(fusion_feature, scene_feature)
        fusion_feature = torch.cat((fusion_feature, scene_feature.unsqueeze(-1)), dim=1)
        fusion_feature = self.channel_attention2(fusion_feature)
        fusion_feature = self.fc3(fusion_feature.squeeze(-1).reshape(-1, 1536))

        return fusion_feature, scene_weight.squeeze(-1)

class ActionClassificationNetwork(nn.Module):
    def __init__(self, num_actions=10):
        super(ActionClassificationNetwork, self).__init__()
        self.fc = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.fc(x)
        return x


class EARNet(nn.Module):
    def __init__(self, num_actions, scene_info=None, batch_size=16):
        super(EARNet, self).__init__()
        self.scene_info = scene_info
        if scene_info:
            self.scene_feature_extraction = ObjectSeparatedAttention(seq_len=64, num_heads=1)
            self.fusion_network = FeatureFusionNetwork()
        self.action_classification_network = ActionClassificationNetwork(num_actions=num_actions)

    def forward(self, action_tensor, scene_tensor1, scene_tensor2):
        scene_weight = None
        fusion_output = None
        if self.scene_info:
            spatial_motion_feature, scene_feature = self.scene_feature_extraction(scene_tensor1.float(), scene_tensor2.float())
            fusion_output, scene_weight = self.fusion_network(action_tensor.float(), spatial_motion_feature, scene_feature)
            action_output = self.action_classification_network(fusion_output)
        else:
            del scene_feature
            action_output = self.action_classification_network(action_tensor.float())
        del action_tensor
        return action_output, scene_weight, fusion_output
