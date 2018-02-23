from pathlib import Path

PATH = Path('..')
RESOURCES = PATH / "resources"
RESOURCES_EN = RESOURCES / "en"
RESOURCES_ES = RESOURCES / "es"
RESOURCES_DE = RESOURCES / "de"
RESOURCES_FR = RESOURCES / "fr"

LMS = {
    "en": RESOURCES_EN / "en.wiki.small.arpa",
    "es": RESOURCES_ES / "es.wiki.arpa",
    "de": RESOURCES_DE / "de.wiki.arpa",
    "fr": RESOURCES_FR / "fr.wiki.arpa"
}

CHAR_LMS = {
    "en": RESOURCES_EN / "en.wiki.small.chars.arpa",
    "es": RESOURCES_ES / "es.wiki.chars.arpa",
    "de": RESOURCES_DE / "de.wiki.chars.arpa",
    "fr": RESOURCES_FR / "fr.wiki.chars.arpa"
}

EMBEDDINGS = {
    "en": RESOURCES_EN / "en.wiki.bpe.op25000.d200.w2v.txt",
    "es": RESOURCES_ES / "es.wiki.bpe.op25000.d200.w2v.txt",
    "de": RESOURCES_DE / "de.wiki.bpe.op25000.d200.w2v.txt",
    "fr": RESOURCES_FR / "fr.wiki.bpe.op25000.d200.w2v.txt"
}
