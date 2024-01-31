from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
import re
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


# to scrape from urls, takes an input array of urls
def url_to_text(inp):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 '
                      'Mobile Safari/537.36'}
    for index, row in inp.iterrows():
        r = requests.get(row['URL'], headers=headers)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, features='lxml')
            articles = soup.find('article', class_='post')
            title = articles.find('h1', class_='entry-title')
            desc = articles.find('div', class_='td-post-content')
            f = open(f'{row["URL_ID"]}.txt', 'w', encoding='utf-8')
            f.write(title.text)
            f.write('\n')
            f.write(desc.text.split('\nBlackcoffer')[0])
            f.close()


# to get words from files provided in local directories
def get_list_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='latin-1') as file:
            words_list = file.read().split()
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: Unable to decode {file_path}", e)

    words_list = [word for word in words_list if word.isalnum()]
    return words_list


# to get stop words from the folder path specified
def get_stop_words(folder_path):
    stop_words_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            stop_words = get_list_from_file(file_path)
            stop_words_dict[filename.split('.txt')[0]] = stop_words

    stopwords_nltk = stopwords.words('english')
    stop_words_dict['StopWords_NLTK'] = stopwords_nltk

    return stop_words_dict


# to create the master dictionary
def get_master_dict(words):
    negative_words = get_list_from_file(r'MasterDictionary\negative-words.txt')
    positive_words = get_list_from_file(r'MasterDictionary\positive-words.txt')
    master = {'positive': [], 'negative': []}
    [master['positive'].append(word) for word in words if word in positive_words and word not in master['positive']]
    [master['negative'].append(word) for word in words if word in negative_words and word not in master['negative']]
    return master


# remove stop words from the text data i.e. data cleaning
def remove_stop_words(text):
    stop_words_dict = get_stop_words(r'StopWords')
    tokens = word_tokenize(text)
    for key, value in stop_words_dict.items():
        tokens = [word for word in tokens if word not in value]
    return tokens


# pretty self-explanatory, does sentiment analysis on text
def sentiment_analysis(text):
    new_text = remove_stop_words(text)
    master = get_master_dict(new_text)
    POSITIVE_SCORE = len(master['positive'])
    NEGATIVE_SCORE = len(master['negative'])
    POLARITY_SCORE = (POSITIVE_SCORE - NEGATIVE_SCORE) / ((POSITIVE_SCORE + NEGATIVE_SCORE) + 0.000001)
    SUBJECTIVITY_SCORE = (POSITIVE_SCORE + NEGATIVE_SCORE) / ((len(new_text)) + 0.000001)

    return POSITIVE_SCORE, NEGATIVE_SCORE, POLARITY_SCORE, SUBJECTIVITY_SCORE


# again, but this one does more than just readability analysis, calculates a few more variables
def readability_analysis(text):
    words = len(text.split())
    sentences = len(text.split('.'))
    syllable_count, complex_words = count_complex_words(text)
    avg_sent_len = words / sentences
    avg_word_len = len(text) / words
    complex_words_per = complex_words / words
    fog_index = 0.4 * (avg_sent_len + complex_words_per)
    syllables_per_word = syllable_count / words
    pronoun_count = count_personal_pronouns(text)

    return [avg_sent_len, complex_words_per, fog_index, avg_sent_len, complex_words, words, syllables_per_word,
            pronoun_count, avg_word_len]


# to count complex words in a text but excluding es/es ending words
def count_complex_words(text):
    # to find words
    word_pattern = r'\b[a-zA-Z-]+\b'

    complex_word_count = 0
    total_syllables = 0
    for word in re.findall(word_pattern, text):

        # to remove common suffixes like "es" and "ed" before counting syllables
        word = re.sub(r'(es|ed)$', '', word, flags=re.IGNORECASE)
        syllable_count = 0
        vowels = "aeiouAEIOU"
        in_word = False
        for char in word:
            if char in vowels:
                if not in_word:
                    syllable_count += 1
                in_word = True
            else:
                in_word = False

        if syllable_count > 2:
            complex_word_count += 1

        total_syllables += syllable_count

    return total_syllables, complex_word_count


def count_personal_pronouns(text):
    # to match the personal pronouns
    pronoun_pattern = r'\b(?:I|we|my|ours|us)\b'

    # to find all matches in the text
    pronoun_matches = re.findall(pronoun_pattern, text)

    # to exclude the word "US" when it refers to the country
    country_reference_pattern = r'\b(?:U\.S\.|US)\b'
    country_reference_matches = re.findall(country_reference_pattern, text)

    # to subtract the count of "US" used in a country reference
    pronoun_count = len(pronoun_matches) - len(country_reference_matches)

    return pronoun_count
