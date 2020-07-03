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
>>> 
>>> text = 'I like reading books more than reading newspaper articles.'
>>> 
>>> summarizer = keywordextraction.Summarizer(preprocess=True)
>>> 
>>> summarizer.tfidf_extract(text)
[[('articles', 0.5773502691896258), ('books', 0.5773502691896258)]]
>>>
>>> summarizer.wordrank_extract(text)
[[('newspaper', 1.1275), ('books', 0.93625), ('articles', 0.93625)]]
>>>
>>> summarizer.fasttext_extract(text)
['articles']
```

In methods tfidf_extract() and wordrank_extract(), the summarizer returns the scores for each keyword. You can ask for as many keywords as there are in the text. Method fasttext_extract() does not return more than one keyword neither any score. These are set aside for future improvements.

## Improvements
- Increase speed of Word2Vec extraction;
- Allow for Word2Vec extraction of multiple keywords;
- Develop new ways of computing sentence embeddings (currently just mean sentence embedding is supported).

## Author
Andre Assumpcao <br>
andre.assumpcao@gmail.com
