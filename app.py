import os
import random

from flask import Flask, request

cryReasons = ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]

app = Flask(__name__)

@app.route('/', methods=["POST","GET"])
def cry_analysis():
    if request.method == "GET":
        cryReason = random.choice(cryReasons)
        return {"cryReason": cryReason}
    else:
        if request.files:
            crySound = request.files['audio']
            print(crySound)
            crySound.save(os.path.join(crySound.filename))
        return ""



if __name__ == '__main__':
    app.run()
