import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=["POST","GET"])
def cry_analysis():
    if request.method == "GET":
        return {"cryReason": "Testing Flask"}
    else:
        if request.files:
            crySound = request.files['audio']
            print(crySound)
            crySound.save(os.path.join(crySound.filename))
        return ""



if __name__ == '__main__':
    app.run()
