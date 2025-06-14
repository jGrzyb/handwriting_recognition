from torch import nn

class HandwritingTransformer(nn.Module):
    def __init__(self, input_size, vocab_size, d_model, nhead_en, num_layers_en, nhead_de, num_layers_de, dropout):
        super(HandwritingTransformer, self).__init__()
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead_en, dropout=dropout),
            num_layers=num_layers_en
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead_de, dropout=dropout),
            num_layers=num_layers_de
        )
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.input_size = input_size
        self.d_model = d_model

    def forward(self, src, tgt):
        src = self.embedding(src) * (self.d_model ** 0.5)
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)
        
        src = self.encoder(src)
        output = self.decoder(tgt, src)
        
        return self.fc_out(output)