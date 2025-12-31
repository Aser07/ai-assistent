import time
import ollama
from langchain_community.utilities import SearxSearchWrapper

# from Voice import Jarvis_answer
from supertonic_mnn import SupertonicTTS
from playsound import playsound


# 1. Initialize
tts = SupertonicTTS()
search = SearxSearchWrapper(searx_host="http://127.0.0.1:8080")
# Models will be downloaded automatically if not present
def create_voice(answer):
    audio, sample_rate = tts.synthesize(f"{str(answer)}", voice="M1", output_file="voice.wav", speed=1.3, steps=7)
    time.sleep(0.3)
    playsound('voice.wav')

# define Tools
def web_search(query):
    # print(f"--- Jarvis is searching for: {query} ---")
    search_result = search.results(str(query), num_results=5)
    return str(search_result)


# tool list
jarvis_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Search the web for real-time information, news, or weather.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The search query to look up on the internet.',
                    },
                },
                'required': ['query'],
            },
        },
    },
]


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
                messages=messages,
                tools=jarvis_tools
            )

            # [단계 2] 모델이 도구 사용을 요청했는지 확인
            if response.get('message', {}).get('tool_calls'):
                for tool in response['message']['tool_calls']:
                    # 함수 이름 확인
                    if tool['function']['name'] == 'web_search':
                        # 중요: 'arguments'를 가져옵니다. (라이브러리에 따라 dict일 수도, str일 수도 있음)
                        args = tool['function'].get('arguments')
                        
                        # 만약 args가 문자열(JSON)로 들어온다면 파싱이 필요할 수 있습니다.
                        # 하지만 최근 ollama 라이브러리는 dict로 반환하는 경우가 많으므로 안전하게 처리:
                        if isinstance(args, str):
                            import json
                            args = json.loads(args)
                        
                        query = args.get('query')
                        
                        if query:
                            search_result = web_search(query)

                            # 검색 결과 피드백을 메시지 기록에 추가
                            messages.append(response['message'])
                            messages.append({
                                'role': 'tool',
                                'content': search_result,
                            })
                        else:
                            print("System Warning: No query found in tool arguments.")

                # [단계 3] 검색 결과를 바탕으로 최종 답변 생성
                final_response = ollama.chat(model='llama3.2:latest', messages=messages)
                assistant_response = final_response['message']['content']
            else:
                # 검색이 필요 없는 일반 대화
                assistant_response = response['message']['content']

            print(f"Jarvis: {assistant_response}")
            create_voice(assistant_response)
            messages.append({'role': 'assistant', 'content': assistant_response})
            

            # Append assistant response to maintain context
            messages.append({'role': 'assistant', 'content': assistant_response})

        except Exception as e:
            print(f"System Error: {e}")

if __name__ == "__main__":
    start_Jarvis_en()