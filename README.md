# custom_word_tokenizer

This Custom Word Tokenizer was used in my master thesis at Gavagai. A word tokenizer to be used in conjunction with the SPACY and NLTK tokenizer. 

This tokenizer wraps the NLTK sentence splitter to avoid splitting sentences that end with ".", "!" or "?" within double or single quotes. Among other small tweaks, it also wraps single words that should not be it's own sentence and it wraps abreviations (e.g. e.g.) and stops the NLTK sentence tokenizer from splitting paragraphs into too many sentences that does not make any sense. 

The Custom Word Tokenizer also includes multiple regular expression rules which orders of operation can be changed to fit the needs of the user. 
