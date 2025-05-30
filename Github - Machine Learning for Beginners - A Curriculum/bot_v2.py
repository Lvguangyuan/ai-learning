import random
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

extractor = ConllExtractor()

# This list contains the random responses (you can add your own or translate them into your own language too)
random_responses = ["That is quite interesting, please tell me more.",
                    "I see. Do go on.",
                    "Why do you say that?",
                    "Funny weather we've been having, isn't it?",
                    "Let's change the subject.",
                    "Did you catch the game last night?"]

print("Hello, I am Marvin, the simple robot.")
print("You can end this conversation at any time by typing 'bye'")
print("After typing each answer, press 'enter'")
print("How are you today?")

while True:
    # wait for the user to enter some text
    user_input = input("> ")
    if user_input.lower() == "bye":
        # if they typed in 'bye' (or even BYE, ByE, byE etc.), break out of the loop
        break
    else:
        user_input_blob = TextBlob(user_input, np_extractor=extractor)
        np = user_input_blob.noun_phrases
        if len(np) > 0:
            response = f"Can you tell me more about {' '.join(np)}. "
        else:
            response = random.choice(random_responses)
        polarity = user_input_blob.polarity
        if polarity <= -0.5:
            sentiment = "Oh dear, that sounds bad. "
        elif polarity <= 0:
            sentiment = "Hmm, that's not great. "
        elif polarity <= 0.5:
            sentiment = "Well, that sounds positive. "
        elif polarity <= 1:
            sentiment = "Wow, that sounds great. "
        response = sentiment + response
    print(response)

print("It was nice talking to you, goodbye!")