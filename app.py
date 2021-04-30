from flask import Flask
from tweet_eval import classify_api

app = Flask(__name__)
app.register_blueprint(classify_api)

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
