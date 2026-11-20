"""Fixed label orders. Do not sort these at runtime."""

# 24 static ASL letters. J and Z are motions, so they live on the LSTM.
STATIC_LABELS: list[str] = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Dynamic vocabulary.
# J and Z are real ASL motion letters. The words are kinematic sequences
# seeded by real ASL handshapes (see signspeak/dynamic_data.py).
DYNAMIC_LABELS: list[str] = [
    "J",
    "Z",
    "HELLO",
    "YES",
    "NO",
    "THANKS",
    "PLEASE",
    "SORRY",
]

# Which static letter supplies the handshape for each dynamic class.
DYNAMIC_HANDSHAPE: dict[str, str] = {
    "J": "I",
    "Z": "D",
    "HELLO": "B",
    "YES": "S",
    "NO": "V",
    "THANKS": "B",
    "PLEASE": "B",
    "SORRY": "A",
}

SEQUENCE_LENGTH = 30
STATIC_EMBED_DIM = 128
DYNAMIC_EMBED_DIM = 32
N_LANDMARKS = 21
