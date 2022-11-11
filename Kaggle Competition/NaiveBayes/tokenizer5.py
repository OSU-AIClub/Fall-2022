import warnings
import re

import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('stopwords')

# Disable warnings for Beautiful Soup Package
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


""" Define Tokenizer - Stemming+ POS Tagging"""
# 0.852875
class Tokenizer:

    def clean(self, text):
        # Remove HTML Tags from input
        no_html = BeautifulSoup(text, 'html.parser').get_text()

        # Replace anything that isn't a letter with a space
        clean = re.sub("[^a-z\s]+", " ", no_html, flags=re.IGNORECASE)

        # Replace all whitespace with a single space
        return re.sub("(\s+)", " ", clean)

    def tokenize(self, text):
        # Clean input text
        clean = self.clean(text)

        stemmer = nltk.stem.SnowballStemmer('english')
        # Use predefined list NLTK's English stopwords
        text = nltk.word_tokenize(clean)
        text = nltk.pos_tag(text)
        text = [(stemmer.stem(w), t) for w, t in text]
        
        
        return text
        
if __name__=='__main__':
    test_sententence = "I think this is a very great and fun test sentence!"
    tokenizer = Tokenizer()
    result = tokenizer.tokenize(test_sententence)
    print(type(result))
    print(result)
