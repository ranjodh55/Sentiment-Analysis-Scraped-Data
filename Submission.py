# main script
import pandas as pd
import helper
import os

# read input and create a dataframe to easily access urls
inp = pd.read_excel('Input.xlsx', dtype={'URL_ID': object})

# create text files from urls
helper.url_to_text(inp)

# get template for output
out = inp.copy()

# populate the output dataframe
for filename in os.listdir():
    out_dict = {}
    if filename.endswith('.txt'):
        f = open(filename, 'r', encoding='latin')
        text = f.read()
        f.close()
        POSITIVE_SCORE, NEGATIVE_SCORE, POLARITY_SCORE, SUBJECTIVITY_SCORE = helper.sentiment_analysis(text)
        readability_scores = helper.readability_analysis(text)
        for index, row in out.iterrows():
            if str(row['URL_ID']) == filename.split('.txt')[0]:
                out.at[index, 'POSITIVE SCORE'] = POSITIVE_SCORE
                out.at[index, 'NEGATIVE SCORE'] = NEGATIVE_SCORE
                out.at[index, 'POLARITY SCORE'] = POLARITY_SCORE
                out.at[index, 'SUBJECTIVITY SCORE'] = SUBJECTIVITY_SCORE
                out.at[index, 'AVG SENTENCE LENGTH'] = readability_scores[0]
                out.at[index, 'PERCENTAGE OF COMPLEX WORDS'] = readability_scores[1]
                out.at[index, 'FOG INDEX'] = readability_scores[2]
                out.at[index, 'AVG NUMBER OF WORDS PER SENTENCE'] = readability_scores[3]
                out.at[index, 'COMPLEX WORD COUNT'] = readability_scores[4]
                out.at[index, 'WORD COUNT'] = readability_scores[5]
                out.at[index, 'SYLLABLE PER WORD'] = readability_scores[6]
                out.at[index, 'PERSONAL PRONOUNS'] = readability_scores[7]
                out.at[index, 'AVG WORD LENGTH'] = readability_scores[8]

# convert the output dataframe back to excel
out.to_excel('Output Data Structure.xlsx')
