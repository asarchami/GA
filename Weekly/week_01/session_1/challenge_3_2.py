
from random import randint
# Here is your random number
random_number = randint(1, 10)

# Number of changes to allow
chances_remaining = 4

# Number of attempts
chances_taken = 0

# Tell the user what they are about to experience
print "Are you excited?  It's time to play guess that number *ENHANCED EDITION*!"

# In a notebook, this will prompt the user to enter a number, and pause execution of the app.
# We use int() because anything that comes into
def play_guess_that_number(random_number, chances_remaining, chances_taken):

    # Increment guess attempt each time
    chances_taken += 1

    ## Your code here
    user_number = int(raw_input("Pick a number between 1-10:  "))
    print "You entered:  ", user_number
    ## Your code here!
    if random_number < user_number and chances_remaining > 0:
        print('too high')
        play_guess_that_number(random_number, chances_remaining - chances_taken, chances_taken)
    elif random_number > user_number and chances_remaining > 0:
        print('too low')
        play_guess_that_number(random_number, chances_remaining - chances_taken, chances_taken)
    elif random_number == user_number and chances_remaining > 0:
        print("You're the winner!!!!!")
    else:
        print("Game Over: Too many guesses!")

# We pass "random_number" so it can be referenced throughout
play_guess_that_number(random_number, chances_remaining, chances_taken)
