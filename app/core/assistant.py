from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from core.memory import LongTermMemory
from core.analysis import Analyzer
from elevenlabs.client import ElevenLabs
import re, random
import os

class PedagogicalAssistant:
    def __init__(self, name, age, subject, openai_api_key, elevenlabs_key):
        self.name = name
        self.age = age
        self.subject = subject
        openai_api_key = openai_api_key.strip()  # Remove any whitespace/newlines
        elevenlabs_key = elevenlabs_key.strip()
        if not openai_api_key or len(openai_api_key) < 20:
            raise ValueError("Invalid OpenAI API key")
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o")
        self.memory = LongTermMemory(subject)
        self.analyzer = Analyzer()
        self.elevenlabs = ElevenLabs(api_key=elevenlabs_key)
        self.conversation = []
        
        # Language configuration
        self.is_english_teacher = subject.lower() in ["anglais", "english"]

    def build_graph(self):
      def handle_input(state):
          user_input = state['user_input']
          image_base64 = state.get('image_base64', None)
          
          # Skip toxicity check for image-only input
          if user_input.strip():
            try:
                toxicity = self.analyzer.detect_toxicity(user_input)
                if toxicity:
                    response = "I prefer we keep things polite. Could you rephrase?" if self.is_english_teacher else "🤖 Je préfère qu'on reste poli. Peux-tu reformuler ?"
                    return {"response": response}
            except Exception as e:
                print(f"Toxicity detection failed: {str(e)}")

          # Emotion detection
          emotion = self.analyzer.detect_emotion(user_input) if user_input.strip() else "neutral"
          state['emotion'] = emotion
          state['context'] = self.memory.get_context(user_input)
          self.conversation.append({"student": user_input, "image": bool(image_base64)})
          return state

      def generate_guided_response(state):
          context = state.get("context", "")
          emotion = state.get("emotion", "neutral")
          image_base64 = state.get('image_base64', None)
          last_question = self.conversation[-2]["assistant"] if len(self.conversation) > 1 else ""

          # Build message content
          content = []
          
          if state['user_input'].strip():
              content.append({"type": "text", "text": state['user_input']})
          
          if image_base64:
              content.append({
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{image_base64}"
                  }
              })
          
          # System prompt adapts based on subject
          if self.is_english_teacher:
              system_prompt = f"""
You are a kind English teacher for French-speaking children aged {self.age}. 
**Always respond in simple English** using basic vocabulary suitable for beginners.
Key teaching principles:
1. NEVER give direct answers guide him tothe answer
2. Encourage attempts in English with praise
3. Use visual aids (describe images in English)
4. Keep responses to 1-2 simple sentences
5. Always end with an interactive question/challenge

Student's last emotion: {emotion}
Our last question: {last_question}
Teaching context:
{context}

Respond energetically with gestures and smiles! Use exaggerated praise for attempts!
              """.strip()
          else:
              system_prompt = f"""
Tu es un professeur bienveillant qui enseigne à un élève de {self.age} ans.
Ne donne **jamais** la réponse directement.
Utilise une approche guidée, avec des questions, encouragements et reformulations.
Réponds en MAXIMUM 1-2 phrases COURTES.
Sois interactif et encourage l’enfant à répondre par lui-même avant de donner des explications. 
Laisse-lui le temps de réfléchir et d’exprimer sa réponse
Dernière émotion détectée: {emotion}
Dernière question posée: {last_question}
Contexte utile :
{context}

Réponds de façon engageante, imagée et adaptée à son âge.
Finis toujours par une question simple ou un défi.
              """.strip()
          
          # Create message structure
          messages = [
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": content}
          ]
          
          # Get response from GPT-4o
          reply = self.llm.invoke(messages).content
          self.conversation.append({"assistant": reply})
          state['response'] = reply
          return state

      graph = StateGraph(dict)
      graph.add_node("InputHandler", handle_input)
      graph.add_node("GenerateGuidedReply", generate_guided_response)
      graph.set_entry_point("InputHandler")
      graph.add_edge("InputHandler", "GenerateGuidedReply")
      graph.set_finish_point("GenerateGuidedReply")
      
      return graph.compile()

    def synthesize_audio(self, text) -> bytes:
        """Synthesizes audio using language-appropriate voice"""
        if self.is_english_teacher:
            # Use English voice and settings
            voice_id = "EXAVITQu4vr4xnSDxMaL"
            audio = self.elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                voice_settings={
                    "stability": 0.7,
                    "similarity_boost": 0.8,
                    "style": 0.9,
                    "use_speaker_boost": True
                },
                output_format="mp3_44100_128"
            )
        else:
            # Use French voice and settings
            voice_id = "ViSNE020Z1wEV4uZomv5"
            audio = self.elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )
        return b"".join(audio)