import torch
import torch.nn as nn
from transformers import AutoModel

class TogetherClassicication(nn.Module) :
    def __init__(self, pre_trained_path, fc_hidden_sizes, dropout_rate) :
        super(TogetherClassicication, self).__init__()

        self.pre_model = AutoModel.from_pretrained(pre_trained_path)

        for param in self.pre_model.parameters() :
            param.requires_grad = False
        
        self.embedding_dim = self.pre_model.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        layers = []
        for i in range(len(fc_hidden_sizes) - 1) :
            layers.append(nn.Linear(fc_hidden_sizes[i], fc_hidden_sizes[i+1]))
            if i < len(fc_hidden_sizes) - 2 : 
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(fc_hidden_sizes[i+1]))
                layers.append(self.dropout)
        self.fc = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask) :
        model_output = self.pre_model(input_ids, attention_mask)

        #sentence_embed = self.mean_pooling(model_output, attention_mask)
        sentence_embed = model_output.pooler_output
        output = self.fc(sentence_embed)
        return output
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
