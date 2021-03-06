
## Q&A Sber Telegram bot 

### Работа с сервисом
Бот доступен в Telegram под именем @gaev_sber_bot
На вход принимает команды /start и текстовые вопросы на тематику клиентского сервиса банка

### Описание содержимого:
* telebot/app.py - стартовый скрипт для авторизации сервиса в Telegram и запуска поллинга сообщений
* telebot/answer_generator.py - класс для обработки сообщения пользователя, его кластеризации и выдачи ближайших вопросов. Класс выгружает LDA модель из NoSQL БД
* telebot/build_model.py - класс для построения LDA модели и сохранения в БД
* telebot/elbow_kmeans.py - построение графика для определения количества кластеров локтевым методом
* telebot/testing.py - проверка модели с вопросами из базы данных, баланс кластеров

### Описание работы сервиса
1) Препроцессинг текста: tokenize, stopwords, links, stemming, lemmatization + сохранение исходной формы слова
2) Векторизация TFIDF и кластеризация LDA моделью с выявлением ключевых слов в кластере
3) Сохранение модели и словарей в БД
4) Генератор сообщений подгружает модель и словари из БД и генерирует ответ в форме [список 5 ближайших троек вида: {вопрос, ответ, confidence}, а также список названий/эталонных вопросов 5 ближайших кластеров с некоторым confidence]
5) Сервис развернут на Heroku который не предоставляет контейнеров для данных, ввиду этого было использована MLab MongoDB для хранения натренированной модели и словарей
6) Подбор числа тематик осуществлен при помощи итераций K means и локтевого метода(Рис. 1 и Рис 2.)
### Ограничения сервиса
* Сервис не суммаризирует термины для кластеров в общую тематику
* Сервис использует BoW модель для векторизации текста, в перспективе может быть рассмотрена word2vec модель, использование n-grams, выявление noun phrase и объекта вопроса
* Необходим более точный метод определения количества кластеров, метод Elbow и K-means дал лишь примерные результаты, можно рассмотреть Model Perplexity and Coherence Score
![alt text](https://raw.githubusercontent.com/romangaev/telebot/master/Figure_1.png)
![alt text](https://raw.githubusercontent.com/romangaev/telebot/master/Figure_2.png)

### Python version
Python 3.6.3
