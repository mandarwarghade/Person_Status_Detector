import xgboost as xgb
import streamlit as st
import pandas as pd
import pickle
#data processing
import re, string
import emoji

#importing nlp library
import nltk

nltk.download('stopwords')
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

#Loading up the Regression model we created


with open('vectorizer_pickle_log','rb') as f:
  model=pickle.load(f)

with open('vectorizer_pickle_log','rb') as f:
  vectorizer=pickle.load(f)




#Caching the model for faster loading
@st.cache_data 


def clean(tp):

    #Clean emojis from text
    def strip_emoji(text):
        #return re.sub(emoji.get_emoji_regexp(), r"", text) #remove emoji
        return emoji.replace_emoji(text, replace='')

    #Remove punctuations, links, mentions and \r\n new line characters
    def strip_all_entities(text): 
        text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
        text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
        text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
        banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
        table = str.maketrans('', '', banned_list)
        text = text.translate(table)
        return text

    #clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
    def clean_hashtags(tweet):
        new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
        new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
        return new_tweet2

    #Filter special characters such as & and $ present in some words
    def filter_chars(a):
        sent = []
        for word in a.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        return ' '.join(sent)

    def remove_mult_spaces(text): # remove multiple spaces
        return re.sub("\s\s+" , " ", text)

    def remove_digit(text):
        result = ''.join([i for i in text if not i.isdigit()])
        return result

    texts_new = []
    for t in df.merged:
        texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(remove_digit(t)))))))

    #Now we can create a new column, for both train and test sets, to host the cleaned version of the tweets' text.
    
    df['text_clean'] = texts_new

    

    #Lowering the text

    df['text_clean']=df['text_clean'].apply(lambda x: x.lower())

    #Stop word removal:

   

    df['text_clean']=df['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

    #Lemmatization: 

   
    #defining the object for Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()

    #defining the function for lemmatization
    def lemmatizer(text):
        lemm_text = "".join([wordnet_lemmatizer.lemmatize(word) for word in text])
        return lemm_text

    df['text_clean']=df['text_clean'].apply(lambda x:lemmatizer(x))

    test=df['text_clean'].values   
    return test


  
# Define the prediction function
def predict(test):
    
    X_test_tf = vectorizer.transform(test)  

    num=model.predict(X_test_tf)[0]

    name={0:'is_patient',1:'is_nurse',2:'is_doctor',3:'is_nonpatient'}

    prediction = name[num]
    return prediction



title_name=['UNKNOWN', 'Personal blog', 'Health  wellness website', 'Healthbeauty', 'Community', 'Coach', 'Entrepreneur', 'Medical  health', 'Nonprofit organization', 'Public figure', 'Doctor', 'Radiologist', 'Surgeon', 'Therapist', 'Personal Coach', 'Designer', 'Writer', 'Digital creator', 'Artist']

st.markdown("<h1 style='text-align: center; color: white;'>status detector</h1>", unsafe_allow_html=True)
#st.image("""img.jpg""")
st.header('Enter the details:')
full_name = st.text_input('Full_name:',)
title=st.selectbox('Title:', title_name)
bio = st.text_area('Bio:')
Post = st.text_area('Post:')
hashtags = st.text_area('Hashtags:')

new_data=" ".join([title, bio,Post,hashtags])

df=pd.DataFrame([new_data])
df.columns=['merged']

test=clean(df) 

if st.button('Predict status of person'):
    value = predict(test)
    st.success(f'The predicted status of person **{full_name}** is **{value}**')
