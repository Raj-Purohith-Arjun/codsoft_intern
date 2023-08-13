import re

def simple_chatbot(user_input):
    user_input = user_input.lower()

    # Define patterns for more complex queries
    greetings = ["hello", "hi", "hey"]
    gratitude = ["thank you", "thanks"]
    inquiry_patterns = {
        r"what.*your name": "I'm a chatbot designed to assist you.",
        r"how.*you.*doing": "I'm just a chatbot, but I'm here and ready to help!",
        r"how.*weather": "I'm sorry, I can't provide real-time information like weather.",
    }

    # Matching patterns
    for pattern, response in inquiry_patterns.items():
        if re.search(pattern, user_input):
            return response

    if any(greeting in user_input for greeting in greetings):
        return "Hi there! How can I assist you today?"
    elif any(thankful in user_input for thankful in gratitude):
        return "You're welcome! If you have more questions, feel free to ask."
    elif "goodbye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I'm not sure how to respond to that."

# Main loop for chatting
print("Simple Chatbot: Hi! I'm a slightly more advanced chatbot. You can start chatting with me. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Simple Chatbot: Goodbye!")
        break
    response = simple_chatbot(user_input)
    print("Simple Chatbot:", response)
