import model_handler
from flask import Flask, request

app = Flask(__name__)
crySoundPath='cry.wav'

@app.route('/', methods=["POST","GET"])
def cry_analysis():
    global crySoundPath
    if request.method == "GET":
        #TODO Get last cry reason from db
        # print(crySoundPath)
        cryReason = model_handler.denoiseAndMakePrediction(crySoundPath)
        return {"cryReason": cryReason}
    else:
        if request.files:
            crySound = request.files['audio']
            crySound.save(crySoundPath)
            cryReason = model_handler.makePrediction(crySoundPath)
            return {"cryReason": cryReason}

if __name__ == '__main__':
    app.run()

