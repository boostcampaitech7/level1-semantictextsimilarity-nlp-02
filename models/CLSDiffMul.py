import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CLSwithDiffMull(nn.Module) :
    def __init__(self, pre_trained_model_path, fc_hidden_sizes, dropout_rate = 0.2) :
        super(CLSwithDiffMull, self).__init__()

        # Pre-trained model load
        self.pre_model = AutoModel.from_pretrained(pre_trained_model_path)

        # Freeze pre-trained model parameters
        for param in self.pre_model.parameters():
            param.requires_grad = False  

        self.embedding_dim = self.pre_model.config.hidden_size
        assert (2 * self.embedding_dim +1) == fc_hidden_sizes[0], \
        "First FC's layer size must same with 4 * embedding_dim after concat"
        assert fc_hidden_sizes[-1] == 1, "Last FC layer's size must 1"

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

        # FC
        layers = []
        for i in range(len(fc_hidden_sizes) - 1) :
            layers.append(nn.Linear(fc_hidden_sizes[i], fc_hidden_sizes[i+1]))
            if i < len(fc_hidden_sizes) - 2 :
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(fc_hidden_sizes[i+1]))
                layers.append(self.dropout)
        self.fc = nn.Sequential(*layers)

    def forward(self, sen1_tokens, sen1_mask, sen2_tokens, sen2_mask) :
        sen1_output = self.pre_model(sen1_tokens, sen1_mask)
        sen2_output = self.pre_model(sen2_tokens, sen2_mask)

        sen1_embed = sen1_output.pooler_output
        sen2_embed = sen2_output.pooler_output
        # sen1_embed = self.mean_pooling(sen1_output, sen1_mask)
        # sen2_embed = self.mean_pooling(sen2_output, sen2_mask)
        # (batch, embedding_dim) 

        # get diff : reflect difference btw 2 sentences
        diff_embed = sen1_embed - sen2_embed
        # get multiply : reflect similarity btw 2 sentences
        mul_embed = sen1_embed * sen2_embed
        cos_sim = F.cosine_similarity(sen1_embed, sen2_embed, dim = -1).unsqueeze(-1)
        # (batch, embedding_dim) 

        # combined = torch.cat((sen1_embed, sen2_embed, diff_embed, mul_embed, cos_sim), dim = -1)
        combined = torch.cat((sen1_embed, sen2_embed, cos_sim), dim = -1)
        
        # (batch, embedding_dim)

        output = self.fc(combined)
        #output = torch.clamp(F.relu(output), min = 0, max = 5)
        #output = torch.sigmoid(output) * 5
        return output
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



