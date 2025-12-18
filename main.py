import uuid
from langchain_core.messages import HumanMessage
from agent import build_agent
from ingestion import ingest_documents

def main():
    print('----- 1. Checking Data Ingestion -----')
    ingest_documents()

    print('----- 2. Starting Chat Agent -----')
    agent = build_agent()

    thread_id = str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}

    print(f'Session ID: {thread_id}')
    print('Type "exit" to quit.\n')

    while True:
        user_input = input('User: ')
        if user_input.lower() in ['exit', 'quit']:
            break

        events = agent.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config,
            stream_mode='values'
        )

        for event in events:
            if 'messages' in event:
                last_msg = event['messages'][-1]
                if last_msg.type == 'ai':
                    print(f'AI: {last_msg.content}')
        print('-' * 50)

if __name__ == '__main__':
    main()