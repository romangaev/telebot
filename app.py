# coding: utf-8
# more examples: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/README.md
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
from answer_generator import AnswerGenerator
import os

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

TG_TOKEN = os.environ['TG_TOKEN']
answer_generator = AnswerGenerator()


def idle_main(bot, update):
    bot.sendMessage(update.message.chat_id, text=answer_generator.generate_answer(update.message.text))
    logging.info("echoing some message...")

def slash_start(bot, update):
    bot.sendMessage(update.message.chat_id, text="Привет! Я - Сбер-бот, бот, который отвечает на вопросы чуть лучше, чем невпопад!")
    logging.info("replying start command...")

def main():
    updater = Updater(TG_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", slash_start), group=0)
    dp.add_handler(MessageHandler(Filters.text, idle_main))
    logging.info("starting polling")
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
