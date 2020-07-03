# Keyword Extraction

## Introduction
This module performs keyword extraction from texts using three alternative methods (TF-IDF, WordRank, and Word2Vec -- on top of FastText word vectors). It extends text summarization tasks into keyword rather than sentence summaries.

## Installation
```
pip install git+https://github.com/aassumpcao/keywordextraction
```

## Usage
```python
>>> import keywordextraction

>>> #first (simpler) example
>>> text = 'I like reading books more than reading newspaper articles.'
>>> summarizer = keywordextraction.Summarizer(preprocess=True)

>>> summarizer.tfidf_extract(text)
[('articles', 0.5773502691896258), ('books', 0.5773502691896258)]]

>>> summarizer.wordrank_extract(text)
[('newspaper', 1.1275), ('books', 0.93625)]

>>> summarizer.fasttext_extract(text)
['articles']

>>> #second (more complex) example
>>> nytimes_wiki = 'The New York Times (sometimes abbreviated as the NYT and NYTimes) is an American newspaper based in New York City with worldwide influence and readership. Founded in 1851, the paper has won 130 Pulitzer Prizes, more than any other newspaper. The Times is ranked 18th in the world by circulation and 3rd in the U.S. Nicknamed "The Gray Lady", the Times has long been regarded within the industry as a national "newspaper of record". The paper\'s motto, "All the News That\'s Fit to Print", appears in the upper left-hand corner of the front page.'

>>> summarizer.tfidf_extract(nytimes_wiki, top_n=5)
[('newspaper', 0.42857142857142855), ('times', 0.42857142857142855), ('new', 0.2857142857142857), ('paper', 0.2857142857142857), ('york', 0.2857142857142857)]

>>> summarizer.wordrank_extract(nytimes_wiki, top_n=5)
[('times', 1.639625), ('newspaper', 1.5199166666666666), ('hand', 1.3612499999999996), ('paper', 1.180625), ('york', 1.1338749999999997)]

>>> summarizer.fasttext_extract(nytimes_wiki)
['newspaper']
```

In methods ```tfidf_extract()``` and ```wordrank_extract()```, the summarizer returns the scores for each keyword. You can ask for as many keywords as there are in the text. You can also set different arguments to the ```Summarizer()``` class as you see fit. Method ```fasttext_extract()``` does not return more than one keyword neither any score. These are set aside for future improvements.

## Improvements
- Increase speed of Word2Vec extraction;
- Allow for Word2Vec extraction of multiple keywords;
- Develop new ways of computing sentence embeddings (currently just mean sentence embedding is supported).

## Author
Andre Assumpcao <br>
andre.assumpcao@gmail.com
