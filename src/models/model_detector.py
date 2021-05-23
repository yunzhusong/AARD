import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

import transformers
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv

class GCNPooler(nn.Module):
    def __init__(self, hidden_size, filter_):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
        self.conv = GCNConv(hidden_size, hidden_size)
        #self.conv2 = GCNConv(hidden_size*2, hidden_size)
        if filter_:
            self.filter = nn.Sequential(
                nn.Linear(2*hidden_size, 1),
                nn.Sigmoid())
        else:
            self.filter = None

    def forward(self, hidden_states, sent_pos=None, edges=None, node_batch=None, append_pos=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token of each sent.
        device = hidden_states.device
        seq_len = hidden_states.size(1)
        if sent_pos is not None and edges is not None:
            sent_pos = self.preprocess_for_append(sent_pos, seq_len, device)
            sent_tensors = hidden_states[sent_pos] # shape (all_sents, hidden_size)

            edge_index, node_batch, weight = self.preprocess_for_GCN(edges, node_batch, device, append_pos)

            if self.filter is not None:
                edge1 = torch.index_select(sent_tensors, 0, edge_index[0])
                edge2 = torch.index_select(sent_tensors, 0, edge_index[1])
                filters = self.filter( torch.cat((edge1, edge2), dim=1) ).view(-1)
                weight = filters * weight

            # layer1
            conv_tensors = self.conv(sent_tensors, edge_index, weight) # shape (all_sents, hidden_size)
            #conv_tensors = F.relu(conv_tensors)
            #conv_tensors = F.dropout(conv_tensors, self.training)
            # layer2
            #conv_tensors = self.conv2(torch.cat((sent_tensors,sent_tensors),dim=1), edge_index)
            #conv_tensors = F.relu(conv_tensors)

            first_token_tensors = scatter_mean(conv_tensors, node_batch, dim=0) # shape (batch_size, hidden_size)
        else:
            first_token_tensors = hidden_states[:, 0] # shape(batch_size, hidden_size)
        pooled_output = self.dense(first_token_tensors)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def preprocess_for_GCN(self, edges, node_batch, device, append_weights):
        """ Combining edges according to correct idx 
            Ex. [[0,0],[1,2]] + [[0,1],[1,2]] -> [[0,0,2,3],[1,2,3,4]] """
        count = 0
        edge_ls, batch_ls, weight_ls = [], [], []
        for i, batch in enumerate(node_batch):
            weight = torch.ones((edges[i].size(1),)).to(device)
            if append_weights is not None:
                new_idx = len(batch) # new idx is the number of old node batch
                batch = torch.cat((batch, batch[0:1])) # append a new node
                append_weight = append_weights[i].squeeze(1)
                num_node = len(append_weight)
                new_edge = torch.tensor([[idx for idx in range(num_node)]
                                         ,[new_idx]*num_node])
                edges[i] = torch.cat((edges[i], new_edge), dim=1)
                weight = torch.cat((weight, append_weight))

            num = len(batch)
            edge = edges[i] + count
            batch = batch + i
            count += num
            batch_ls.append(batch)
            edge_ls.append(edge)
            weight_ls.append(weight)

        return torch.cat(edge_ls, dim=1).to(device),\
               torch.cat(batch_ls).to(device), \
               torch.cat(weight_ls).to(device)

    def preprocess_for_append(self, sent_pos, seq_len, device):
        bs, old_seq_len = sent_pos.shape
        new_sent_len = seq_len - old_seq_len
        if new_sent_len == 0:
            return sent_pos

        temp = torch.zeros(bs, new_sent_len)
        temp[:,0] = 1
        new_pos = temp==1

        return torch.cat((sent_pos, new_pos.to(device)), dim=1)


class GCNClassifier(nn.Module):
    """
    """
    def __init__(self, hidden_dropout_prob, hidden_size, num_labels, filter_):
        super(GCNClassifier, self).__init__()

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.pooler = GCNPooler(hidden_size, filter_)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.apply(self._init_weights)
        #self._init_weights(self.classifier)


    def forward(self, outputs, edges=None, node_batch=None, append_pos=None, 
                labels=None, return_dict=True):
 
        if edges is not None and node_batch is not None:
            pooled_output = self.pooler(outputs.last_hidden_state, 
                                        outputs.sent_pos,
                                        edges=edges, 
                                        node_batch=node_batch, 
                                        append_pos=append_pos)
        else:
            pooled_output = self.pooler(outputs.last_hidden_state)

        #pooled_output = outputs.pooled_output

        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02) # std: initializer_range
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


