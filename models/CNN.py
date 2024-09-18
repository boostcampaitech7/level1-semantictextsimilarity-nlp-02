import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class CNNWithEmbedding(nn.Module) :
    def __init__(self, pre_trained_model_path, num_filters = 100, 
                 filter_sizes = [3,4,5], fc_sizes = [300, 150, 1], dropout_rate = 0.5) :
        super(CNNWithEmbedding, self).__init__()

        # Pre-trained BERT load
        self.bert = AutoModel.from_pretrained(pre_trained_model_path)
        self.embedding_dim = self.bert.config.hidden_size

        # CNN Layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, self.embedding_dim)) for fs in filter_sizes
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)

        assert (len(filter_sizes) * num_filters) == fc_sizes[0], "First FC layer's size must same with last CNN's output size"
        assert fc_sizes[-1] == 1, "Last FC layer's size must 1"
        layers = []
        for i in range(len(fc_sizes)-1) :
            layers.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))
            if i < len(fc_sizes) -2 :
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(fc_sizes[i+1]))
                layers.append(self.dropout)
        self.fc = nn.Sequential(*layers)

    def forward(self, sen1, sen2) :
        # Don't apply mask for now.
        
        # get embedding vector using pre-trained model
        embedded_sen1 = self.get_bert_embeddings(sen1).unsqueeze(1)
        embedded_sen2 = self.get_bert_embeddings(sen2).unsqueeze(1)
        # (batch, 1, num_tokens, embedding_dim)

        # CNN 
        conved_sen1 = [F.relu(self.batch_norms[i](conv(embedded_sen1))).squeeze(3) for i, conv in enumerate(self.convs)]
        conved_sen2 = [F.relu(self.batch_norms[i](conv(embedded_sen1))).squeeze(3) for i, conv in enumerate(self.convs)]
        # (batch, num_filters, after_cnn, 1) -> (batch, num_filters, after_cnn)

        # Max-pooling
        pooled_sen1 = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conved_sen1]
        pooled_sen2 = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conved_sen2]
        # (batch, num_filters, 1) -> (batch, num_filters)

        # Concat each sentence
        concat_sen1 = torch.cat(pooled_sen1, dim = 1)
        concat_sen2 = torch.cat(pooled_sen2, dim = 1)
        # (batch, num_filters * len(filter_sizes))

        # Add two sentence
        combined = concat_sen1 + concat_sen2
        combined = self.dropout(combined)

        # Fully connected layer
        output = self.fc(combined).unsqueeze(1)
        output = 5 * F.sigmoid(output)
        # (batch, num_filters * len(filter_sizes)) -> (batch, 1)

        return output
    
    def get_bert_embeddings(self, sentences) :
        embeddings = self.bert(sentences)['last_hidden_state'] 
        # (batch, num_tokens, embedding_dim)
        return embeddings