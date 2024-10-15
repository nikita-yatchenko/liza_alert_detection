# Human Detection


## Как запустить инференс: 
- Заполняем файл .env, пример можно посмотреть в [.env.example](.env.example)
-  Создаем виртуальное окружение и активируем его:
```
python3 -m venv venv
source venv/bin/activate
```
- Устанавливаем зависимости:
```
pip3 install poetry
poetry install
```
- Запускаем сервис: 
```
python src/solution.py < example_predict.json
```

## Как запустить через docker:
```
docker build . -t detection
docker run -i detection < example_predict.json  
```
