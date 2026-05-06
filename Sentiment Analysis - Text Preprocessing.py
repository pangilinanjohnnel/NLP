import pandas as pd
import nltk
from symspellpy import SymSpell
import pkg_resources
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer

tr=pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\D5.1\archive\test.csv")

#for spelling
ss = SymSpell(max_dictionary_edit_distance=2)
path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
ss.load_dictionary(path, 0, 1)

#for stop words and stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

#for chunker
grammar = "NP: {<JJ>*<NN.*>+}"
rp = nltk.RegexpParser(grammar)

#for cleaning
def clean_text(text):
    text = str(text).lower() #lower
    tokens = word_tokenize(text) #tokenize
    clean = [w for w in tokens if w.isalpha() and w not in stop_words] #alpha only
    spell = [] #spelling
    for w in clean:
        s = ss.lookup(w, 1, 2)
        spell.append(s[0].term if s else w)
    correct = [w for w in spell if w]
    pos = nltk.pos_tag(correct) #pos
    stem = [(ps.stem(word), tag) for word, tag in pos] #stemming
    return stem

#for chunking
def extract_chunks(tagged_list):
    tree = rp.parse(tagged_list)
    return [" ".join(w for w, t in leaf) for leaf in tree.subtrees() if leaf.label() == "NP"]

tr["text"] = tr["text"].apply(clean_text)
tr["chunk"] = tr["text"].apply(extract_chunks)

#for top 10
all_words = [word for row in tr["text"] for word, tag in row]
print("\n--- TOP 10 WORDS ---")
for word, count in Counter(all_words).most_common(10):
    print(word, ":", count)