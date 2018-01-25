import glob

# DATASET FORMAT DEFINITON
ID = 0
SENTENCE = 1
START = 2
END = 3
TARGET = 4
NATIVE_SEEN = 5
FOREIGN_SEEN = 6
NATIVE_COMPLEX = 7
FOREIGN_COMPLEX = 8
LABEL_ANY = 9
LABEL_FRACTION = 10

DATASET_FIELDS_TRAIN = [ID, SENTENCE, START, END, TARGET, NATIVE_SEEN,
                        FOREIGN_SEEN, NATIVE_COMPLEX, FOREIGN_COMPLEX,
                        LABEL_ANY, LABEL_FRACTION]
DATASET_FIELDS_TEST = [ID, SENTENCE, START, END, TARGET, NATIVE_SEEN,
                       FOREIGN_SEEN]

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


def get_data(lang, split):
    # dataset = Dataset()
    dataset = []
    dataset_files = glob.glob("../data/{}/*{}.tsv".format(lang, split))
    print("Reading files: ", dataset_files)
    for df in dataset_files:
        with open(df) as f:
            for line in f:
                fields = line.strip().split(FIELD_SEPARATOR)
                if split in ["Train", "Dev"]:
                    assert len(fields) == len(DATASET_FIELDS_TRAIN), \
                        "Different field numbers ({} vs {})".\
                            format(len(fields), len(DATASET_FIELDS_TRAIN))
                elif split in ["Test"]:
                    assert len(fields) == len(DATASET_FIELDS_TEST), \
                        "Different field numbers ({} vs {})".\
                            format(len(fields), len(DATASET_FIELDS_TRAIN))
                dataset.append(fields)
    return dataset
