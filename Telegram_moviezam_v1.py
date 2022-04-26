#!/usr/bin/env python
# coding: utf-8

# In[4]:
"""attention  :  pip install python-telegram-bot --upgrade    NOT pip install telegram"""

# import libraries
import pandas as pd
from gensim.models import Word2Vec
import pickle
import emoji
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import simplemma
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import logging
from typing import Dict

from telegram import ReplyKeyboardMarkup, Update, ReplyKeyboardRemove,InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,CallbackQueryHandler,
 
)



# upload w2v model
model_w2v = pd.read_pickle('pickle_w2v')
X_word2vec = pd.read_pickle('pickle_w2v_vectors')

# prepare dataframe
data_movielens_final_new=pd.read_excel('data_movielens_final_new.xlsx')
data_movielens_final_new['ID_Length']=data_movielens_final_new['ID'].astype(str).apply(len)
data_movielens_final_new['zeros']=np.where(data_movielens_final_new['ID_Length']==7, '', np.where(data_movielens_final_new['ID_Length']==6,'0',np.where(data_movielens_final_new['ID_Length']==5,'00',np.where(data_movielens_final_new['ID_Length']==4,'000',np.where(data_movielens_final_new['ID_Length']==3,'0000',np.where(data_movielens_final_new['ID_Length']==2,'00000','000000'))))))
data_movielens_final_new['Links'] = 'https://www.imdb.com/title/tt'+data_movielens_final_new['zeros']+data_movielens_final_new['ID'].astype(str)

# functions for search and reccomandation
def search_plot_similarity_adjusted(text,encoding_method,encoded_corpus,dataframe,topk,model=None,vectorizer=None,features=None):
    TOKEN_LEMMA = [clean_text_lemma(text)]
    space_query=encoding_method(TOKEN_LEMMA,model)
    cosineSimilarities = cosine_similarity(space_query, encoded_corpus).flatten()
    df=dataframe.iloc[np.argsort(cosineSimilarities)[-topk*20:].tolist() ]
    df['similarity']=np.sort(cosineSimilarities)[-topk*20:].tolist() 
    df['final_score']=(df['similarity']*0.87)+(df['votes_score']*0.13)
    df=df.sort_values(['final_score'], ascending=False)
    df=df.groupby('Title').first().reset_index()
    df=df.nlargest(topk,'final_score')
    
    df=df[['Title','Links']]

    return df
def word2vec_encoding_query(text,model_w2v,percentage_of_vocabulary_to_use=None, min_count=5):
    size=300
    corpus = []
    tokenizer = WordPunctTokenizer()
    for sent in text:
        corpus.append(tokenizer.tokenize(sent.lower().translate(str.maketrans('', '', string.punctuation))))
    vec = np.zeros(size).reshape((1, size))
    wordvec_arrays = np.zeros((len(corpus), 300)) 
    for i in range(len(corpus)):
        count = 0
        for word in corpus[i]:
            try:
                vec += model_w2v.wv[word].reshape((1, size)) 
                count += 1.
            except KeyError:  # handling the case where the token is not in vocabulary
                continue
        if count != 0:
            vec /= count
        wordvec_arrays[i,:] = vec
    
    return   wordvec_arrays 
stopwords_en = stopwords.words('english')
langdata = simplemma.load_data('en')

def clean_text_lemma(text):
    if type(text) == float:
        return ""
    temp = text.lower() 
    temp = re.sub("'", " ", temp) # in inglese rimuove le parole abbreviate ma anche in italiano serve a staccare le parola con apostrofo
    temp=emoji.demojize(temp, language='en')
    words = temp.split()
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9èéàùòì]"," ", temp)
    temp = re.sub(" \d+", " ", temp)
    temp = re.sub("^[0-9]", " ", temp)
    #temp = temp.split()
    temp = WordPunctTokenizer().tokenize(temp)
    temp = [simplemma.lemmatize(w, langdata) for w in temp if not w in stopwords_en]
    temp = " ".join(word for word in temp)
    
    return temp


def search_movie_similarity_adjusted(title,encoding_method,encoded_corpus,dataframe,topk,model=None,vectorizer=None,features=None):
    df=dataframe
     # ratio method is more accurate , but slower
    df['sim'] = [difflib.SequenceMatcher(a=title.lower(), b=i.lower()).ratio() for i in df['Title']   ]   
    df=df.sort_values(['sim'], ascending=False)
    df1=df.head(1)
    title_found=df1['Title'].item()
    df2=df[df['Title']==title_found]
    df2=df2.dropna(subset=['text_clean'])
    df2['doc_len'] = df2['text_clean'].apply(lambda words: len(words.split()))
    df2=df2.sort_values(['doc_len'], ascending=False)
    df2=df2.head(1)
    df2=df2.drop(['sim'], axis=1)
    query_new=df2['text_clean'].item()
    query_new

#
    # print("Most similar to:" + title_found)
    # print("Top " +str(topk)+ " titles, by plot similarity")
    df_new=search_plot_similarity_adjusted(text=query_new,encoding_method=encoding_method,encoded_corpus=encoded_corpus,dataframe=dataframe,topk=topk,model=model,vectorizer=vectorizer,features=features)
    df_new=df_new[df_new['Title']!=title_found] 
    df_new['plot_searched']=query_new
    df_new['Title_to_search']=title_found
    cols = df_new.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_new = df_new[cols]
    return df_new





# here start the telegram chatbot
from logging import basicConfig, getLogger, INFO

basicConfig(level=INFO)
log = getLogger()


# Enable logging
#logging.basicConfig(
 #   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
#)
TOKEN='5389959641:AAFzapxb0HqNfNtEKCBrpVJzTx5AexVX5Bk'

# State definitions for top level conversation
CHOOSING,QUERY_PLOT,QUERY_TITLE,WHATDONOW = map(chr, range(4))
# Shortcut for ConversationHandler.END
END = ConversationHandler.END

# Callback data
ONE, TWO,THREE,FOUR = range(4)



def start(update: Update, context: CallbackContext) -> int:
    """Start the conversation and ask user for input."""
    

    keyboard = [
        [
            InlineKeyboardButton("Search by plot events", callback_data=str(ONE)),
            InlineKeyboardButton("Suggest movies", callback_data=str(TWO)),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    

    update.message.reply_text(
        
        "Hi  ! I'm" "* Moviezam!* \U0001F916" " I know everything about""*  Movies!* \U0001F4FD" "\n" 
        "Do you want to search for movies with" "* a particular plot event?*" " \n" 
        "Can i ""*suggest you similar movies *""to the ones you like?\n" 
         "Beware: my catalog is updated " "*until 2019*"  "\U0001F622" "\n" 
         "To abort, simply type /stop.",
        #   reply_markup=reply_markup,parse_mode= 'Markdown',
    parse_mode= 'Markdown',
    )
    
    
    
    update.message.reply_text(

        "*Choose one*"" of the options below, I will do a search for you \U0001F603 !" "\n",
        #   reply_markup=reply_markup,parse_mode= 'Markdown',
        reply_markup=reply_markup ,parse_mode= 'Markdown',
    )
    

    return CHOOSING

def answer_recommend(update: Update, context: CallbackContext) -> int:
    """Ask the user for info about the selected predefined choice."""
    query = update.callback_query
    query.answer()
    query.edit_message_text(
        text="Please write the *title* of a movie that you like, i will suggest you 5 similar movies \n"
        "Beware that the movie you submit is the same i found in my database",parse_mode= 'Markdown'
    )

    return QUERY_TITLE




def stop(update: Update, context: CallbackContext) -> int:
    """End Conversation by command."""
    update.message.reply_text('Okay, bye! \U0001F603'  "\n"
                             "If you want to start again click or type /start")

    return END

def answer_plot(update: Update, context: CallbackContext) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    query.answer()
    query.edit_message_text(
        text="Please write some *plot events or details* and i will show you some related movies",parse_mode= 'Markdown'
    )
    return QUERY_PLOT

def echo_plot(update: Update, context: CallbackContext) -> None:
        
    text_to_search=update.message.text
    df=search_plot_similarity_adjusted(text=text_to_search,encoding_method=word2vec_encoding_query,model=model_w2v,encoded_corpus=X_word2vec,dataframe=data_movielens_final_new,topk=10)
    df_lista=df['Links'].tolist()
    df_lista_titles=df['Title'].tolist()
    update.message.reply_text(f'Movie 1: {df_lista_titles[0]}')                         
    update.message.reply_text(df_lista[0])
    update.message.reply_text(f'Movie 2: {df_lista_titles[1]}') 
    update.message.reply_text(df_lista[1])
    update.message.reply_text(f'Movie 3: {df_lista_titles[2]}') 
    update.message.reply_text(df_lista[2])
    update.message.reply_text(f'Movie 4: {df_lista_titles[3]}') 
    update.message.reply_text(df_lista[3])
    update.message.reply_text(f'Movie 5: {df_lista_titles[4]}') 
    update.message.reply_text(df_lista[4])
    
    
    keyboard = [
        [
            InlineKeyboardButton("Start over again", callback_data=str(THREE)),
            InlineKeyboardButton("I'm done for now", callback_data=str(FOUR)),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    

    update.message.reply_text(
        
           "What do you want to do, now? ",
        #   reply_markup=reply_markup,parse_mode= 'Markdown',
        reply_markup=reply_markup ,parse_mode= 'Markdown',
    )
    
    
    return WHATDONOW


def echo_title(update: Update, context: CallbackContext) -> None:
    text_to_search=update.message.text
    update.message.reply_text('Wait please... \U0001F605 ') 
    df=search_movie_similarity_adjusted(title=text_to_search,encoding_method=word2vec_encoding_query,model=model_w2v,encoded_corpus=X_word2vec,dataframe=data_movielens_final_new,topk=10)
    df_lista=df['Links'].tolist()
    df_lista_titles=df['Title'].tolist()
    title_found=df['Title_to_search'].tolist()
    title_found=title_found[0]
    update.message.reply_text(f'You submit the movie title : {text_to_search}') 
    update.message.reply_text(f'I found in my database the following movie title : {title_found}')  
    update.message.reply_text(f'Below you can find 5 most similar movies to:  {title_found}')  
    update.message.reply_text(f'Movie 1: {df_lista_titles[0]}')                         
    update.message.reply_text(df_lista[0])
    update.message.reply_text(f'Movie 2: {df_lista_titles[1]}') 
    update.message.reply_text(df_lista[1])
    update.message.reply_text(f'Movie 3: {df_lista_titles[2]}') 
    update.message.reply_text(df_lista[2])
    update.message.reply_text(f'Movie 4: {df_lista_titles[3]}') 
    update.message.reply_text(df_lista[3])
    update.message.reply_text(f'Movie 5: {df_lista_titles[4]}') 
    update.message.reply_text(df_lista[4])
    
    
    keyboard = [
        [
            InlineKeyboardButton("Start over again", callback_data=str(THREE)),
            InlineKeyboardButton("I'm done for now", callback_data=str(FOUR)),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    

    update.message.reply_text(
        
           "What do you want to do, now? ",
        #   reply_markup=reply_markup,parse_mode= 'Markdown',
        reply_markup=reply_markup ,parse_mode= 'Markdown',
    )
    
    
    return WHATDONOW

def startover(update: Update, context: CallbackContext) -> None:
    """Start the conversation and ask user for input."""
    

    keyboard = [
        [
            InlineKeyboardButton("Search by plot events", callback_data=str(ONE)),
            InlineKeyboardButton("Suggest movies", callback_data=str(TWO)),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query = update.callback_query
    query.answer()
    query.edit_message_text(
        "*Choose one*"" of the options below, I will do a search for you  \U0001F603 ! \n"
         "To abort, simply type /stop.",
        #   reply_markup=reply_markup,parse_mode= 'Markdown',
        reply_markup=reply_markup ,parse_mode= 'Markdown',
    )


    return CHOOSING


def stop2(update: Update, context: CallbackContext) -> int:
    
    
    """End Conversation by command."""
    
    query = update.callback_query
    query.answer()
    query.edit_message_text(
       (                 'Okay, bye! \U0001F603' "\n"
                             "If you want to start again click or type /start"),
        #   reply_markup=reply_markup,parse_mode= 'Markdown',
        
    )

    return END



def main() -> None:
    """Run the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(token=TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING: [
                 CallbackQueryHandler(answer_plot, pattern='^' + str(ONE) + '$'),
                 CallbackQueryHandler(answer_recommend, pattern='^' + str(TWO) + '$'),
              #  MessageHandler(
               #     Filters.regex('^(Search movie by plot events)$'), answer_plot   ),
                #MessageHandler(Filters.regex('^Recommend movies by similarity$'), answer_recommend),
            ],
            
            
            QUERY_PLOT: [MessageHandler(Filters.text & ~Filters.command, echo_plot)],
            QUERY_TITLE: [MessageHandler(Filters.text & ~Filters.command, echo_title)],
                
             WHATDONOW: [
                 CallbackQueryHandler(startover, pattern='^' + str(THREE) + '$'),
                 CallbackQueryHandler(stop2, pattern='^' + str(FOUR) + '$'),
              #  MessageHandler(
               #     Filters.regex('^(Search movie by plot events)$'), answer_plot   ),
                #MessageHandler(Filters.regex('^Recommend movies by similarity$'), answer_recommend),
            ],
       
        },
        
        fallbacks=[CommandHandler('stop', stop)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling(timeout=100)

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()

