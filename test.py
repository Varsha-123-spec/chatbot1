import joblib 
model=joblib.load('model.pkl')
while True:
    user_input = input("varshini")
    if user_input.lower() == 'exit':
        break
    response = model.predict([user_input])
    print("chatbot", response[0])
    