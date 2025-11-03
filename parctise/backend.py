from flask import Flask, request, Response, render_template_string, render_template
from search_2 import *

app = Flask(__name__,template_folder="forefront")
@app.route("/")
def index():
    return render_template("index.html")

# 生成器：每次返回一个字符


@app.route("/api/ask")
def ask():
    question = request.args.get("question", "")
    return Response(chat(question), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True)