import contractions
import numpy as np
import re, spacy
from scipy.spatial.distance import cdist
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict

class Summarizer:

    """
        creates a text summarizer object for creating synthetic
        classifications of occupations or industries
    """

    def __init__(
            self, preprocess=False, stopwords=None, pos=['NOUN', 'PROPN'],
             d=0.85, min_diff=1e-5, steps=10, node_weight=None
        ):

        # define class objects
        self.preprocess = preprocess
        self.stopwords = stopwords
        self.pos = pos
        self.d = d
        self.min_diff = min_diff
        self.steps = steps
        self.node_weight = node_weight
        self.nlp = spacy.load('en_core_web_lg')

    # create internal method for processing text
    def _preprocess(self, text):
        processed = text.lower().strip()
        processed = contractions.fix(processed)
        processed = re.split(r' +', processed)
        processed = [re.sub(r'\'s', '', element) for element in processed]
        processed = [re.sub(r'[^\w \-]', '', element) for element in processed]
        processed = list(filter(None, processed))
        return ' '.join(processed)

    # create internal method for deleting stopwords
    def _remove_stopwords(self, text):
        processed = re.split(r' ', text)
        processed = [word for word in processed if not word in self.stopwords]
        return ' '.join(processed)

    # create method for tf-idf extraction
    def tfidf_extract(self, texts, top_n=2, **kwargs):
        if not isinstance(texts, list):
            texts = [texts]
        if self.preprocess:
            texts = [self._preprocess(text) for text in texts]
        if self.stopwords:
            texts = [self._remove_stopwords(text) for text in texts]
        if not self.pos:
            self.pos = list(spacy.glossary.GLOSSARY.keys())[:20]

        # create nlp pipe
        self.docs = self.nlp.pipe(texts, batch_size=len(texts), n_process=-1)

        sentences = []
        for doc in self.docs:
            tokens = [token for sent in doc.sents for token in sent]
            tokens = [
                token.text for token in tokens
                if token.pos_ in self.pos and not token.is_stop
            ]
            tokens = ' '.join(tokens)
            sentences += [tokens]
        texts = sentences.copy()

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)
        vocab = vectorizer.get_feature_names()
        scores = []
        for vector in vectors:
            sentence = []
            for _, idx, score in zip(*sparse.find(vector)):
                sentence.append((vocab[idx], score))
            scores.append(sentence)

        kwargs = {'key': lambda x: x[1], 'reverse': True}
        scores = [sorted(score, **kwargs) for score in scores]
        scores = [score[:top_n] for score in scores]

        if len(scores) == 1:
            return scores[0]
        else:
            return scores

    # create internal functions for wordrank extraction
    def _sentence_segment(self, text, lower=True):
        sentences = []
        for sent in text.sents:
            selected_words = []
            for token in sent:
                if token.pos_ in self.pos and not token.is_stop:
                    if lower:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    # get all tokens available
    def _get_vocab(self, sentences):
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    # build token_pairs from windows in sentences
    def _get_token_pairs(self, window_size, sentences):
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    # make symmetric matrix
    def _symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    # get normalized matrix
    def _get_matrix(self, vocab, token_pairs):

        # build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # get symmeric matrix
        g = self._symmetrize(g)

        # normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0)

        return g_norm

    # create method for WordRank extraction
    def _wordrank_extract(self, text, top_n window_size=2, lower=True):

        # create nlp object
        doc = self.nlp(text)

        # main function to analyze text
        if self.stopwords:
            # set stop words
            for word in self.stopwords:
                lexeme = self.nlp.vocab[word]
                lexeme.is_stop = True

        # filter sentences
        sentences = self._sentence_segment(doc, lower)

        # build vocabulary
        vocab = self._get_vocab(sentences)

        # get token_pairs from windows
        token_pairs = self._get_token_pairs(window_size, sentences)

        # get normalized matrix
        g = self._get_matrix(vocab, token_pairs)

        # initialization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # get weight for each node
        node_weight = []
        for word, index in vocab.items():
            node_weight += [(word, pr[index])]

        kwargs = {'key': lambda x: x[1], 'reverse': True}
        node_weight = sorted(node_weight, **kwargs)
        scores = [score[:top_n] for score in node_weight]
        return scores

    # create external method to report wordrank
    def wordrank_extract(self, texts, top_n=2):

        # transform texts in list
        if not isinstance(texts, list):
            texts = [texts]

        # extract keywords
        output = [
            self._wordrank_extract(text, top_n) for text in texts
        ]

        if len(output) == 1:
            return output[0]
        else:
            return output

    # format vocab to use in distance function
    def _prepare_spacy_vocab(self):
        ids = [x for x in self.nlp.vocab.vectors.keys()]
        vectors = [self.nlp.vocab.vectors[x] for x in ids]
        vectors = np.array(vectors)
        return ids, vectors

    # create method to produce words closest to mean sentence embedding
    #  of titles / description
    def fasttext_extract(self, texts):

        # convert text object to process one or multiple text object
        if not isinstance(texts, list):
            texts = [texts]

        # process text if asked for in class
        if self.preprocess:
            texts = [self._preprocess(text) for text in texts]
        if self.stopwords:
            texts = [self._remove_stopwords(text) for text in texts]
        if not self.pos:
            self.pos = list(spacy.glossary.GLOSSARY.keys())[:20]

        # create nlp pipe
        self.docs = self.nlp.pipe(texts, batch_size=len(texts), n_process=-1)

        # call spacy vocab
        ids, vectors = self._prepare_spacy_vocab()

        # create nlp sentences object
        sentences = []
        for doc in self.docs:
            sentence = [token for token in doc]
            sentences.append(sentence)

        # process sentences object and vectors
        processed_sentences, processed_vectors = [], []
        for sentence in sentences:
            sentence_text, sentence_vector = [], []
            for token in sentence:
                if token.pos_ in self.pos:
                    sentence_text.append(token.text)
                    sentence_vector.append(token.vector)
            processed_sentences.append(sentence_text)
            processed_vectors.append(sentence_vector)

        #filter non-empty elements of text
        idx = [i for i, token in enumerate(processed_sentences) if token]
        processed_sentences = [processed_sentences[i] for i in idx]
        processed_vectors = [processed_vectors[i] for i in idx]

        # compute mean sentence embedding and increase their dimension by one
        embeddings = [np.mean(vector, axis=0) for vector in processed_vectors]
        embeddings = [embedding[np.newaxis, :] for embedding in embeddings]

        # compute closest words to sentence embeddings
        words = []
        for embedding in embeddings:
            indexes = cdist(embedding, vectors, 'cosine')
            closest = indexes.argmin()
            word_id = ids[closest]
            keyword = self.nlp.vocab[word_id].text.lower()
            words.append(keyword)

        # return words
        return words
