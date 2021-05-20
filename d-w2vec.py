# doc2vec
#https://github.com/jhlau/doc2vec

# word2vec
from gensim.models import word2vec
# sentences = word2vec.Text8Corpus('./data/text8')
# model = word2vec.Word2Vec(sentences, size=512)
# model.save('text8.model')
model = word2vec.Word2Vec.load('text8.model')
print(len(model['have']))


def conceptword():
    import nltk
    concept = {}
    with open(r'.\data\train_vocab_en.txt', encoding='utf-8') as f:
        vocab = [word.strip() for word in f.readlines()]
    for i in range(4,len(vocab)):
        tokens = nltk.word_tokenize(vocab[i])
        text = nltk.Text(tokens)
        tags = nltk.pos_tag(text)
        for c, j in tags:
            # if j in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and len(c)>1:
            if j in ['NN', 'VB'] and len(c) > 1:
                concept[i]=c
    return concept


concept=conceptword()
concept_feature={}
for i,j in concept.items():
    try:
        concept_feature[i]=model[j]
    except:
        pass

print(len(concept_feature))
