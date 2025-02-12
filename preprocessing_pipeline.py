import string
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    return tokens

# Generate WordCloud
def generate_wordcloud(tokens, ngram=1):
    if ngram == 1:
        # Unigram WordCloud
        text_data = ' '.join(tokens)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
    else:
        # Bigram WordCloud using CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=list(stop_words))
        X = vectorizer.fit_transform([' '.join(tokens)])
        bigrams = vectorizer.get_feature_names_out()
        bigram_freq = dict(zip(bigrams, X.toarray().sum(axis=0)))

        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(bigram_freq)

    return wordcloud
