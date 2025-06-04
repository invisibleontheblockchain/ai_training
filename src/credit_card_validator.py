def luhn_checksum(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = 0
    checksum += sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10

def get_card_type(card_number):
    if card_number.startswith('4'):
        return 'Visa'
    elif card_number.startswith(('51', '52', '53', '54', '55')):
        return 'Mastercard'
    elif card_number.startswith(('34', '37')):
        return 'Amex'
    else:
        return 'Unknown'

def format_card_number(card_number):
    return ' '.join([card_number[i:i+4] for i in range(0, len(card_number), 4)])

def validate_email(email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

if __name__ == "__main__":
    card_number = input("Enter a credit card number: ")
    if is_luhn_valid(card_number):
        print(f"Card number {card_number} is valid.")
        card_type = get_card_type(card_number)
        print(f"Card type: {card_type}")
        formatted_card = format_card_number(card_number)
        print(f"Formatted card number: {formatted_card}")
    else:
        print(f"Card number {card_number} is invalid.")
    email = input("Enter an email address: ")
    if validate_email(email):
        print(f"Email address {email} is valid.")
    else:
        print(f"Email address {email} is invalid.")
