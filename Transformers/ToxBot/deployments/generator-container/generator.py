import torch
from gtts import gTTS
import argparse

import os
import keyboard
import time

from tqdm import tqdm
import random
import speech_recognition as sr
import fastapi

app = fastapi.FastAPI()

date = time.time()


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    print("cold mic...")

    # set up the response object
    response = {"success": True, "error": None, "transcription": None}

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return (
        response
        if response["success"]
        else recognize_speech_from_mic(recognizer, microphone)
    )


def generate(model, tokenizer, context):
    # tokenize question and text as a pair
    encodings = tokenizer.encode_plus(context)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    output = model.generate(
        inputs=torch.tensor([input_ids]),
        attention_mask=torch.tensor([attention_mask]),
        do_sample=True,
        num_beams=3,
        max_new_tokens=50,
        temperature=1.8,
        repetition_penalty=1.32,
    )
    return tokenizer.decode(output[0])


def speak(text):
    myobj = gTTS(text=text, lang="en", slow=False)
    myobj.save(f"output/speech-{date}.mp3")
    # Playing the converted file
    return os.system(f"mpg321 output/speech-{date}.mp3")


def sample_context(path):
    with open(path, "r") as contexts:
        lines = contexts.readlines()
        return random.sample(lines, 1)[0]


@app.get("/run/")
def run(context: str):
    try:
        print(context)
        idx_start = len(context)

        # Model
        with open("models/generator.pth", "rb") as f:
            model = torch.load(f, map_location=torch.device("cpu"))

        # Tokenizer
        with open("models/tokenizer.pth", "rb") as f:
            tokenizer = torch.load(f, map_location=torch.device("cpu"))

        # Create output file
        with open("output/crazy.txt", "w") as crazy:
            # Generate while reseeding 5 times
            for _ in tqdm(range(5)):
                context = generate(model, tokenizer, str(context).strip())
                context = (
                    context.replace("\n", "")
                    .replace("<|endoftext|>", "")
                    .replace("<pad>", "")
                )
            # Write completed generation to file
            crazy.write(context)
            # Display response
            print(context[idx_start:])
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"Blew up: {e}")


def main():
     # Model
    with open("../../models/generator.pth", "rb") as f:
        model = torch.load(f, map_location=torch.device("cpu"))

    # Tokenizer
    with open("../../models/tokenizer.pth", "rb") as f:
        tokenizer = torch.load(f, map_location=torch.device("cpu"))

    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        print("\n\n\n\n\n\n\n\n\n\n\n\nReady.")
        if keyboard.read_key() == "space":
            # context = recognize_speech_from_mic(recognizer, microphone)["transcription"]
            context = sample_context("../../data/context.txt")
            idx_start = len(context)

            # Create output file
            with open(f"../../output/crazy-{date}.txt", "w") as crazy:
                # Generate while reseeding 5 times
                for _ in tqdm(range(5)):
                    context = generate(model, tokenizer, str(context).strip())
                    context = (
                        context.replace("\n", "")
                        .replace("<|endoftext|>", "")
                        .replace("<pad>", "")
                    )
                # Write completed generation to file
                crazy.write(context)
                # Display response
                print(context[idx_start:])
                # Create and play an audio file of the output
                speak(context[idx_start:])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    if args.test:
        main()
    else:
        import uvicorn

        uvicorn.run(app, host="localhost", port=8001)
