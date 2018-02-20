from pathlib import Path

RANDOM_SEED = 42

PATH = Path('..')
RESOURCES = PATH / "resources"
RESOURCES_EN = RESOURCES / "en"
RESOURCES_ES = RESOURCES / "es"
RESOURCES_DE = RESOURCES / "de"
RESOURCES_FR = RESOURCES / "fr"

LMS = {
    "en": RESOURCES_EN / "lm_giga_64k_nvp_2gram.arpa"
}

EMBEDDINGS = {
    "en": RESOURCES_EN / "en.wiki.bpe.op25000.d200.w2v.txt",
    "es": RESOURCES_ES / "es.wiki.bpe.op25000.d200.w2v.txt",
    "de": RESOURCES_DE / "de.wiki.bpe.op25000.d200.w2v.txt",
    "fr": RESOURCES_DE / "fr.wiki.bpe.op25000.d200.w2v.txt"
}
