import pandas as pd
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import logging
import pickle
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model

from string import punctuation
from stop_words import get_stop_words
from pymorphy2 import MorphAnalyzer

path = '../../../data/spam_detection/'


exclude = set(punctuation)
sw = set(get_stop_words("ru"))
morpher = MorphAnalyzer()

def preprocess_text(txt):
    txt = str(txt)
    txt = "".join(c for c in txt if c not in exclude)
    txt = txt.lower()
    txt = re.sub("\sне", "не", txt)
    txt = [morpher.parse(word)[0].normal_form for word in txt.split() if word not in exclude]
    return " ".join(txt)


df = pd.read_pickle(os.path.join(path, 'df_processed.pkl'))
print(df.shape)
df.head(3)

pkl_path_filename = os.path.join(path, 'lr_model.pkl')
with open(pkl_path_filename, 'rb') as file:
    lr = pickle.load(file)
lr

pkl_path_filename = os.path.join(path, 'count_vectorizer.pkl')
with open(pkl_path_filename, 'rb') as file:
    count_vect = pickle.load(file)
count_vect


def check_spam(text):
    text = preprocess_text(text)
    return lr.predict(count_vect.transform(pd.Series(text)))


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO
                    )

logger = logging.getLogger()

def start(update: Update, context: CallbackContext):
    update.message.reply_text('Привет!')

def echo(update: Update, context: CallbackContext):
    msg = update.message.text
    update.message.reply_text('Ваше сообщение: ' + msg)
    
    if check_spam(msg) == 1:
        update.message.reply_text('Это спам!')
    else:
        update.message.reply_text('Это не спам')

with open(os.path.join(path, 'token.txt'), "r") as file:
    token = file.read()


updater = Updater(token=token, use_context=True) 
dispatcher = updater.dispatcher


dispatcher.add_handler(CommandHandler("start", start))

dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

updater.start_polling(drop_pending_updates=True)

updater.idle()
