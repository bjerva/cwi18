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

def sent_to_pos(nlp, fields):
    tagged = nlp(fields[1])
    char_start, char_end = int(fields[2]), int(fields[3])
    words = fields[1][char_start:char_end].split()
    tags = [word.pos_ for word in tagged if word.text in words]

    return(tags)

if __name__ == '__main__':
    for fname in argv[1:]:
        lang = fname.split('/')[2]
        print(fname, lang)
        nlp = spacy.load(lang, disable=['parser', 'ner', 'textcat'])
        with open(fname, 'r', encoding='utf-8') as in_f:
            with open(fname+'.pos', 'w', encoding='utf-8') as out_f:
                for line in in_f:
                    fields = line.split('\t')
                    pos = sent_to_pos(nlp, fields)
                    out_f.write(line.strip())
                    out_f.write('\t{0}\n'.format(' '.join(pos)))
