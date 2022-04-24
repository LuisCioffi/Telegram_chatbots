#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""Readme :  pip install python-telegram-bot --upgrade"""

import logging
from typing import Dict

from telegram import ReplyKeyboardMarkup, Update, ReplyKeyboardRemove
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
TOKEN='5389959641:AAFzapxb0HqNfNtEKCBrpVJzTx5AexVX5Bk'

# State definitions for top level conversation
CHOOSING = map(chr, range(1))
# Shortcut for ConversationHandler.END
END = ConversationHandler.END

reply_keyboard = [
    [ 'Search movie by plot events','Recommend movies by similarity'],]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)



def start(update: Update, context: CallbackContext) -> int:
    """Start the conversation and ask user for input."""
    update.message.reply_text(
        
        "Hi  ! I'm Moviezam! I know everything about movies! \n" 
        "Do you want to search for movies with a particular plot event? \n" 
        "Can i suggest you similar movies to the ones you like?\n" 
        "Choose on of the options below, I start searching for you\n"
         "To abort, simply type /stop.",
        reply_markup=markup,
    )

    return CHOOSING

def answer_recommend(update: Update, context: CallbackContext) -> int:
    """Ask the user for info about the selected predefined choice."""
    text = update.message.text
    context.user_data['choice'] = text
    update.message.reply_text('Please write the title of a movie that you like, i will suggest you 5 similar movies')

    return TYPING_REPLY

def answer_plot(update: Update, context: CallbackContext) -> int:
    """Ask the user for info about the selected predefined choice."""
    text = update.message.text
    context.user_data['choice'] = text
    update.message.reply_text('Please write some plot events or details and i will show you some related movies')

    return TYPING_REPLY

def stop(update: Update, context: CallbackContext) -> int:
    """End Conversation by command."""
    update.message.reply_text('Okay, bye.')

    return END


def main() -> None:
    """Run the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING: [
                MessageHandler(
                    Filters.regex('^(Search movie by plot events)$'), answer_plot   ),
                MessageHandler(Filters.regex('^Recommend movies by similarity$'), answer_recommend),
            ],
       
        },
        
        fallbacks=[CommandHandler('stop', stop)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()

