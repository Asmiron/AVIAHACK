FROM python:3.9
# Установите рабочий каталог в контейнере
WORKDIR /app
# Копируйте файлы в контейнер
COPY . /app
# Установите необходимые зависимости
RUN pip install -r requirements.txt
# Укажите команду для запуска приложения
CMD ["python", "train.py"]
