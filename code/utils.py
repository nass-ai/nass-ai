from gensim.models.doc2vec import TaggedDocument


def handle_format(text_list, train=True):
    output = []
    for index, value in enumerate(text_list):
        tag = f"train_{index}" if train else f"test_{index}"
        output.append(TaggedDocument(value.split(), [tag]))
    return output
