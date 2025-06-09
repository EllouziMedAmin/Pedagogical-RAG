from transformers import pipeline

class Analyzer:
    def __init__(self):
        self.toxicity = pipeline("text-classification", model="unitary/toxic-bert")
        self.emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

    def detect_toxicity(self, text):
        results = self.toxicity(text)
        print(results)
        return any(label['label'] == 'toxic' and label['score'] > 0.5 for label in results)

    def detect_emotion(self, text):
        print(self.emotion(text)[0][0]['label'])
        return self.emotion(text)[0][0]['label']