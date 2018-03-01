import glob

# DATASET FORMAT DEFINITON
ID = 0
SENTENCE = 1
START = 2
END = 3
TARGET = 4
NATIVE_SEEN = 5
FOREIGN_SEEN = 6

TARGET_SENT_SIMILARITY = 7
N_NOUN = 8
N_VERB = 9
N_ADJ = 10
N_ADV = 11
N_ADP = 12
N_PROPN = 13
N_NUM = 14

NATIVE_COMPLEX = 15
FOREIGN_COMPLEX = 16
LABEL_ANY = 17
LABEL_FRACTION = 18

# this is for non-augmented data
NATIVE_COMPLEX_ORIG = 7
FOREIGN_COMPLEX_ORIG = 8
LABEL_ANY_ORIG = 9
LABEL_FRACTION_ORIG = 10

DATASET_FIELDS_TRAIN = [ID, SENTENCE, START, END, TARGET, NATIVE_SEEN,
                        FOREIGN_SEEN, NATIVE_COMPLEX, FOREIGN_COMPLEX,
                        LABEL_ANY, LABEL_FRACTION]

DATASET_FIELDS_TEST = [ID, SENTENCE, START, END, TARGET, NATIVE_SEEN,
                       FOREIGN_SEEN]

EXTENDED_FIELDS = [TARGET_SENT_SIMILARITY, N_NOUN, N_VERB, N_ADJ, N_ADP, N_ADV,
                   N_PROPN, N_NUM]

FIELD_SEPARATOR = "\t"


class Dataset(list):
    """
    Class holding a dataset. In its current state just a wrapper round a
    list of lists... `instances` is a list of dataset examples, each of which
    is again a list of values for the fields defined above
    """
    def __init__(self, instances=None):
        self.instances = instances
        super().__init__()

    def add_instance(self, instance):
        self.instances.append(instance)

    def __len__(self):
        if type(self.instances) == list:
            return len(self.instances)
        return 0


def get_data(lang, split, augmented=True):
    # dataset = Dataset()
    dataset = []
    path = "../data/{}/*{}.tsv"
    expected_length = len(DATASET_FIELDS_TEST) if split == "Test" else \
        len(DATASET_FIELDS_TRAIN)
    if augmented:
        path = "../data_augmented/{}/*{}.tsv"
        expected_length += len(EXTENDED_FIELDS)
    dataset_files = glob.glob(path.format(lang, split))
    print("Reading files: ", dataset_files)
    for df in dataset_files:
        with open(df) as f:
            for line in f:
                fields = line.strip().split(FIELD_SEPARATOR)
                assert len(fields) == expected_length, \
                    "Different field numbers ({} vs {})".format(
                        len(fields), expected_length)
                dataset.append(fields)
    return dataset


def write_out(predictions, filename):
    if predictions.dtype == bool:
        predictions = predictions.astype(int)
    with open(filename, "w") as out:
        out.write("\n".join([str(p) for p in predictions]))
