import re
from collections import defaultdict

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Disable warnings for Beautiful Soup Package
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


""" Define Tokenizer w/ Lemmatization and POS Tagging"""
# 85.9%
class Tokenizer:
  tag_map = defaultdict(lambda: wn.NOUN)
  tag_map['J'] = wn.ADJ
  tag_map['V'] = wn.VERB
  tag_map['R'] = wn.ADV
  
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
      
      tokens = [lemmatizer.lemmatize(word, self.tag_map[tag[0]]) for word, tag in pos_tag(nltk.word_tokenize(clean)) if word not in stopwords_en]
      # Return the list of tokens (i.e. tokenized text)
      return tokens
  

