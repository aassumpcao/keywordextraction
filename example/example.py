import keywordextraction

#first (simpler) example
text = 'I like reading books more than reading newspaper articles.'
summarizer = keywordextraction.Summarizer(preprocess=True)

#run methods
summarizer.tfidf_extract(text)
summarizer.wordrank_extract(text)
summarizer.fasttext_extract(text)

#second (more complex) example
nytimes_wiki = 'The New York Times (sometimes abbreviated as the NYT and NYTimes) is an American newspaper based in New York City with worldwide influence and readership. Founded in 1851, the paper has won 130 Pulitzer Prizes, more than any other newspaper. The Times is ranked 18th in the world by circulation and 3rd in the U.S. Nicknamed "The Gray Lady", the Times has long been regarded within the industry as a national "newspaper of record". The paper\'s motto, "All the News That\'s Fit to Print", appears in the upper left-hand corner of the front page.'

#run methods
summarizer.tfidf_extract(nytimes_wiki, top_n=5)
summarizer.wordrank_extract(nytimes_wiki, top_n=5)
summarizer.fasttext_extract(nytimes_wiki)


