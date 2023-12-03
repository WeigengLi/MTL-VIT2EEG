import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

'''
    Classical Transformer with cos PE
'''


class Spatial_Temporal_Transformer_v7(nn.Module):
    def __init__(self, input_dim=129, max_len=500, output_dim=2, d_model=768, num_heads=12, num_encoder_layers=12,
                 num_decoder_layers=12):
        super(Spatial_Temporal_Transformer_v7, self).__init__()

        # Linear layer to transform input dimension
        self.input_linear = nn.Linear(input_dim, d_model)

        # Using a pre-trained BERT model as the encoder
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.hidden_size = d_model
        config.num_hidden_layers = num_encoder_layers
        self.bert_encoder = BertModel(config)

        # Decoder part of the Transformer
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_decoder_layers
        )

        # Predictor for eye position
        self.predictor = torch.nn.Sequential(torch.nn.Linear(d_model, d_model // 2, bias=True),
                                             torch.nn.Dropout(p=0.3),
                                             torch.nn.Linear(d_model // 2, d_model // 4, bias=True),
                                             torch.nn.Dropout(p=0.3),
                                             torch.nn.Linear(d_model // 4, output_dim, bias=True))

        # Linear layer to transform decoder output back to input dimension
        self.output_linear = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # Input Shape: [batch_size, 1, input_dim, seq_len]

        # Shape: [seq_len, batch_size, feature_dim]
        x = x.squeeze(1).transpose(0, 2).transpose(1, 2)
        # Shape: [seq_len, batch_size, d_model]
        x = self.input_linear(x)

        # Encoder output using BERT
        x = self.bert_encoder(inputs_embeds=x).last_hidden_state

        # Branch 1: Predicting eye position
        position_pred = self.predictor(x[-1, :, :])

        # Branch 2: Reconstructing input
        reconstructed_x = self.decoder(x, x)
        reconstructed_x = self.output_linear(reconstructed_x)
        reconstructed_x = reconstructed_x.transpose(1, 2).transpose(0, 2).unsqueeze(1)

        return position_pred, reconstructed_x


if __name__ == '__main__':
    # Instantiate the model
    model = Spatial_Temporal_Transformer_v7()

    # Create a dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 129, 500)  # Using random values

    # Forward pass
    positions, x_reconstructed = model(input_tensor)

    # Print output shapes to verify
    print("Positions Shape:", positions.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)
