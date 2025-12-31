import time
import ollama

# from Voice import Jarvis_answer
from supertonic_mnn import SupertonicTTS
from playsound import playsound

# 1. Initialize
tts = SupertonicTTS()

# 2. Synthesize
# Models will be downloaded automatically if not present
def create_voice(answer):
    audio, sample_rate = tts.synthesize(f"{str(answer)}", voice="M1", output_file="voice.wav", speed=1.3, steps=7)
    time.sleep(0.3)
    playsound('voice.wav')

def start_Jarvis_en():
    # Context management with English instructions
    messages = [
        {
            'role': 'system', 
            'content': "You are Jarvis. Sophisticated, British, and polite. Always address the user as 'Sir'. IMPORTANT: Keep your responses extremely concise and to the point. Do not be wordy."
        }
    ]

    greeting_messege = "--- Jarvis is Online. (Type 'exit' to shut down.) ---"
    exit_mssege = "Always a pleasure working with you, Sir. Powering down."
    print(greeting_messege)
    create_voice(greeting_messege)

    while True:
        # User input
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "shut down"]:
            print(exit_mssege)
            create_voice(exit_mssege)
            break

        # Append user message
        messages.append({'role': 'user', 'content': user_input})

        try:
            # Generate response from Ollama
            response = ollama.chat(
                model='llama3.2:latest', # Llama3 performs excellently in English
                messages=messages
            )

            # Process assistant response
            assistant_response = response['message']['content']
            # Jarvis_answer(str(assistant_response))/
            print(f"Jarvis: {assistant_response}")
            create_voice(assistant_response)
            

            # Append assistant response to maintain context
            messages.append({'role': 'assistant', 'content': assistant_response})

        except Exception as e:
            print(f"System Error: {e}")

if __name__ == "__main__":
    start_Jarvis_en()