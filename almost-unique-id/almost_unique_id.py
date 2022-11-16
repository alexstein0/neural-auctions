import random

from adjectives import adjectives
from names import names


def generate_id():
    hashstr = f"{adjectives[random.randint(0, len(adjectives))]}-{names[random.randint(0, len(names))]}"
    return hashstr


if __name__ == "__main__":
    print(f"{len(adjectives) * len(names)} Unique IDs possible.")
