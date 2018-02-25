#!/usr/bin/env python

'''
Run example:
python src/scripts/augment_with_pos.py ./data_augmented/*/*

Requires models for de,en,es,(fr):
sudo python -m spacy download <lang>

Note: slightly bugged
'''

import spacy
from sys import argv
from util.io import *
from collections import Counter
from features.functions import TargetSentenceSimilarity
from scipy import spatial
from tqdm import tqdm
from features.featurize import featurize
from util.io import get_data
import numpy as np


def sent_to_pos(nlp, fields):
    tagged = nlp(fields[SENTENCE])
    start, end = int(fields[START]), int(fields[END])
    tags = [word.pos_ for word in tagged if word.idx in range(start, end)]
    return (tags)

langs = [
    "en",
    "de",
    "es",
    # "fr"
         ]

splits = [
    "Train",
    "Dev"
]

for lang in langs:
    for split in splits:
        data = get_data(lang, split, augmented=False)
        nlp = spacy.load(lang, disable=['parser', 'ner', 'textcat'])
        tss = TargetSentenceSimilarity(language=lang)
        sims, _ = featurize(data, [tss], scale_features=False, augmented=False)
        sims = np.array(sims).reshape(-1)
        print(sims.shape)
        print("Writing out...")
        with open("../data_augmented/{}/{}.tsv".format(lang, split), 'w',
                  encoding='utf-8') as out_f:
            for i, fields in enumerate(tqdm(data)):
                sim = sims[i]
                pos = sent_to_pos(nlp, fields)
                c = Counter(pos)
                out = fields[:TARGET_SENT_SIMILARITY] + \
                      [str(sim)[:7], c['NOUN'], c['VERB'], c['ADJ'], c['ADV'],
                       c['ADP'], c['PROPN'], c['NUM']] + \
                      fields[TARGET_SENT_SIMILARITY:]
                out_f.write("\t".join([str(f) for f in out])+"\n")
