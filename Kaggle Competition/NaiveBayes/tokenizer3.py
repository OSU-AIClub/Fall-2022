import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('omw-1.4')

# Disable warnings for Beautiful Soup Package
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


""" Define Tokenizer w/ Lemmatization"""
# 86.25%
class Tokenizer:
  
  def clean(self, text):
      # Remove HTML Tags from input
      no_html = BeautifulSoup(text, 'html.parser').get_text()
      
      # Replace all whitespace with a single space and normalize to lower case
      return re.sub("(\s+)", " ", no_html).lower()

 
  def tokenize(self, text):
      # Clean input text
      clean = self.clean(text).lower()
      
      # Use predefined list NLTK's English stopwords
      stopwords_en = stopwords.words("english")
      lemmatizer = WordNetLemmatizer()
      
      # Return the list of tokens (i.e. tokenized text)
      return [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(clean) if w not in stopwords_en]
  

