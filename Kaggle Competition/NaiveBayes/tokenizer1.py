import re

import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('stopwords')

# Disable warnings for Beautiful Soup Package
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

""" Define Tokenizer w/out Removing Non-letteres"""
# Acuracy: 0.8515
class Tokenizer:
  
  def clean(self, text):
      # Remove HTML Tags from input
      no_html = BeautifulSoup(text, 'html.parser').get_text()
      
      # Replace all whitespace with a single space
      return re.sub("(\s+)", " ", no_html)

 
  def tokenize(self, text):
      # Clean input text
      clean = self.clean(text).lower()
      
      # Use predefined list NLTK's English stopwords
      stopwords_en = stopwords.words("english")
      
      # Return the list of tokens (i.e. tokenized text)
      return [w for w in re.split("\W+", clean) if not w in stopwords_en]
  

