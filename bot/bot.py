!pip install aiogram -q
!python -m spacy download ru_core_news_sm
import spacy
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec
import json
import numpy as np

# загружаем SpaCy модель для русского языка
nlp = spacy.load("ru_core_news_sm")

# инициализация бота
dp = Dispatcher()
bot = Bot(token='my_token')

# функция для лемматизации текста с использованием spaCy
def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

# загружаем FAQ из JSON
!wget https://raw.githubusercontent.com/vifirsanova/compling/main/tasks/task3/faq.json
with open('faq.json', encoding='utf-8') as f:
    data = json.load(f)

# извлекаем вопросы и ответы
faq_questions = [q['question'] for i in data.values() for q in i]
faq_answers = [q['answer'] for i in data.values() for q in i]

# лемматизируем вопросы из FAQ
faq_questions_lemmatized = [lemmatize_text(q) for q in faq_questions]

# преобразование вектора для TF-IDF (используем лемматизированные вопросы)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_questions_lemmatized)
# инициализация модели Word2Vec
sentences = [q.split() for q in faq_questions_lemmatized]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)


# функция для получения ответа с использованием TF-IDF
def get_tfidf_answer(question):
    # лемматизируем запрос
    lemmatized_question = lemmatize_text(question)
    # преобразуем запрос в вектор
    query_vec = vectorizer.transform([lemmatized_question])
    # вычисляем косинусное сходство
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    # ищем индекс наиболее похожего вопроса
    best_match_idx = similarities.argmax()
    return faq_answers[best_match_idx]

# функция для усреднения векторов слов в вопросе
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) # берем среднее значение по всем векторам, чтобы одно предложение представлял один вектор

# векторизуем вопросы
faq_vectors = np.array([sentence_vector(query, word2vec_model) for query in faq_questions_lemmatized])

# функция для получения ответа с использованием Word2Vec
def get_word2vec_answer(question):
    # лемматизируем запрос
    lemmatized_question = lemmatize_text(question)
    # генерируем вектор запроса, усреднив векторы всех слов
    query_vector = sentence_vector(lemmatized_question, word2vec_model).reshape(1, -1)
    # оценка косинусного сходства с каждым вопросом из FAQ
    similarities = cosine_similarity(query_vector, faq_vectors)
    best_match_idx = similarities.argmax()

    return faq_answers[best_match_idx]

# обработка команды /start
@dp.message(Command("start"))
async def welcome(message: types.Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[ [KeyboardButton(text='О компании')], [KeyboardButton(text='Пожаловаться')] ],
        resize_keyboard=True
    )
    await message.answer('Привет! Я бот, помогу ответить на вопросы.', reply_markup=keyboard)

# обработка нажатия кнопки 'О компании'
@dp.message(lambda message: message.text == 'О компании')
async def about_company(message: types.Message):
    await message.answer('Наша компания занимается доставкой товаров по всей стране.')

# обработка нажатия кнопки 'Пожаловаться'
@dp.message(lambda message: message.text == 'Пожаловаться')
async def complain(message: types.Message):
    await message.answer('Отправьте скриншот проблемы для обработки жалобы.')

# обработка изображений
@dp.message(lambda message: message.content_type == "photo")
async def get_photo(message: types.Message):
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    filename = file.file_path.split("/")[-1]
    filesize = message.photo[0].file_size
    await message.answer(f'Название файла: {filename}\nРазмер файла: {filesize} байт\nВаш запрос передан специалисту')

# обработка вопросов пользователей
@dp.message()
async def answers(message: types.Message):
    question = message.text
    # ответ на основе TF-IDF
    tfidf_answer = get_tfidf_answer(question)
    # ответ на основе Word2Vec
    word2vec_answer = get_word2vec_answer(question)
    # отправляем оба ответа пользователю
    await message.answer(f"Ответ на основе TF-IDF: {tfidf_answer}")
    await message.answer(f"Ответ на основе Word2Vec: {word2vec_answer}")


# основной цикл
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    await main()
