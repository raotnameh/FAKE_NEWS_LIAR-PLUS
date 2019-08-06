import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class BoWClassifier(nn.Module): 

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        
        self.linear1 = nn.Linear(vocab_size, 256)
        self.linear2 = nn.Linear(256, num_labels)
        self.relu = nn.ReLU6(True)
        self.drop = nn.Dropout(p=0.5, inplace=False)


    def forward(self, bow_vec):
        out = self.relu(self.linear1(bow_vec))
        out = self.drop(out)
        out = (self.linear2(out))
       
        return F.log_softmax(out, dim=1)
    
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
    

class EncoderRNN(nn.Module):
    def __init__(self,embedding, hidden_size, num_layers,directions,bidirectonal,out):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.directions = directions
        self.embedding, num_embeddings, embedding_dim = embedding
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True,bidirectional=bidirectonal)
        self.linear1 = nn.Linear(hidden_size*directions, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, out)
        self.drop = nn.Dropout(p=0.5, inplace=False)

    def forward(self, inp, hidden):
        out = self.embedding(inp)
        out,h = self.gru(out,hidden)
#         print(out.shape)
#         print(out)
        out = self.drop(out)
        out = self.linear1(out[-1][-1].view(1,-1))
        out = self.drop(out)
        out = self.linear2(out)
        out = self.drop(out)
        out = self.linear3(out)
        return F.log_softmax(out,dim=1)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers*self.directions, batch_size, self.hidden_size)
