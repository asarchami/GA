# You may enjoy writing your own random number function, but for the sake of time, let's use Python's built-in random module.
# from random import randint
#
# # Here is your random number
# random_number = randint(1, 10)
#
# # Tell the user what they are about to experience
# print "Are you excited?  It's time to play guess that number!"
#
# # In a notebook, this will prompt the user to enter a number, and pause execution of the app.
# # We use int() because anything that comes into
# def play_guess_that_number(random_number):
#
#         user_number = int(raw_input("Pick a number between 1-10:  "))
#             print "You entered:  ", user_number
#                 ## Your code here!
#                     if random_number > user_number:
#                             print('too high')
#                                     play_guess_that_number(random_number)
#                                         elif random_number < user_number:
#                                                 print('too low')
#                                                         play_guess_that_number(random_number)
#                                                             else:
#                                                                     print("You're the winner!!!!!")
#
#                                                                     # We pass "random_number" so it can be referenced throughout
#                                                                     play_guess_that_number(random_number)
