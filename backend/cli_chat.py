from conversation_manager import ConversationManager

def main():
    print("FitBot CLI - type 'quit' to exit\n")
    manager = ConversationManager()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("FitBot: ", end="", flush=True)
        manager.send_message(user_input)
        print()


if __name__ == "__main__":
    main()