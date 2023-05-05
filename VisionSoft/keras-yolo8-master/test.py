import random
import string

def generate_key(length):
    letters_and_digits = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

key = generate_key(32)
print(key)
