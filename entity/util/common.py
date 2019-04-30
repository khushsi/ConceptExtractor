from entity.util import nlp
import pickle

def getConcepts(filename):
    gold_list = set()

    with open(filename) as f:
        for line in f:
            gold_list.add(' '.join(
                nlp.preprocessText(line.replace("\n", "").strip().lower())))

    return list(gold_list)


def load_model(filename):
    model = pickle.load(open(filename,'rb'))
    return model
