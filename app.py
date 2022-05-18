import os
import random
import model_handler

from flask import Flask, request

#cryReasons = ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]


app = Flask(__name__)
crySoundPath='.\\cry.wav'

@app.route('/', methods=["POST","GET"])
def cry_analysis():
    global crySoundPath
    if request.method == "GET":
        #TODO Get last cry reason from db
        print(crySoundPath)
        cryReason = model_handler.makePrediction(crySoundPath)
        return {"cryReason": cryReason}
    else:
        if request.files:
            crySound = request.files['audio']

            crySoundPath=".\\"+os.path.join(crySound.filename)

            crySound.save(crySoundPath)
            #TODO Import the crySound to the ML model below
            #TODO Store the cry reason in the db
        return ""


if __name__ == '__main__':
    app.run()
