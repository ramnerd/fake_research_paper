import torch
from torch import nn
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

# --- INIT PARAMETERS ---
vocab_size = 10000
max_len = 128
d_model = 64
num_heads = 4
drop_prob = 0.1
ffn_hidden = 128
num_classes = 2
num_layers = 5

# --- SAMPLE DATA (Fake vs Real Abstracts) ---
texts = [
    "This paper introduces a novel technique using 5D tensor wormholes in GPU algorithms.",  # FAKE
    "We present a new image classification method based on attention mechanisms.",           # REAL
    "Quantum dragons are used to compress transformer networks.",                            # FAKE
    "Our study explores the effects of adversarial learning on translation accuracy."        # REAL
]
labels = [0, 1, 0, 1]  # 0 = Fake, 1 = Real

# --- TOKENIZATION ---
token = Tokenizer(num_words=vocab_size, oov_token="<unk>")
token.fit_on_texts(texts)
seqs = token.texts_to_sequences(texts)
padded = pad_sequences(seqs, maxlen=max_len, padding='post')

X = torch.tensor(padded, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- EMBEDDING & POSITIONAL ENCODING ---
emb = nn.Embedding(vocab_size, d_model)
pos = nn.Embedding(max_len, d_model)

# Position IDs
position_ids = torch.arange(max_len).unsqueeze(0).repeat(X_train.size(0), 1)

# Embed Input + Position
x = emb(X_train) + pos(position_ids)

# --- MHA + FFN + DROPOUT + LAYERNORM ---
mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

ffn = nn.Sequential(
    nn.Linear(d_model, ffn_hidden),
    nn.ReLU(),
    nn.Dropout(drop_prob),
    nn.Linear(ffn_hidden, d_model)
)

drop1 = nn.Dropout(drop_prob)
drop2 = nn.Dropout(drop_prob)

ln1 = [nn.LayerNorm(d_model) for _ in range(num_layers)]
ln2 = [nn.LayerNorm(d_model) for _ in range(num_layers)]

# --- ENCODER LAYER FUNCTION ---
def encoder_layer(x, ln1, ln2):
    # --- Multi-Head Attention ---
    residual = x
    attn_out, _ = mha(x, x, x, need_weights=False)
    x = drop1(attn_out)
    x = ln1(x + residual)

    # --- Feed Forward Network ---
    residual = x
    ffn_out = ffn(x)
    x = drop2(ffn_out)
    x = ln2(x + residual)

    return x

# --- APPLY ENCODER STACK ---
for i in range(num_layers):
    x = encoder_layer(x, ln1[i], ln2[i])

# --- MEAN POOLING & CLASSIFICATION ---
pooled = x.mean(dim=1)  # Average over sequence length
classifier = nn.Linear(d_model, num_classes)
logits = classifier(pooled)

print("\nLogits:\n", logits)
print("Predictions:", torch.argmax(logits, dim=1))
print("Ground Truth:", y_train)
