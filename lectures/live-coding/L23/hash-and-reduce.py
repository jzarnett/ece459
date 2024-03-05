"""Rainbow table operates on 7-digit numbers.

$ python hash-and-reduce.py 1234567
Plaintext:  1234567
MD5sum:     fcea920f7412b5da7be0cf42b8c93759
Reduced:    9207412

"""

from hashlib import md5
from sys import argv


def reduce(digest):
    """Reduce a hash into a new 7-digit number."""
    reduced = ""
    for c in digest:
        if c.isdigit():
            reduced = reduced + c
            if len(reduced) == 7:
                return reduced


plaintext = argv[1]
md5_digest = md5(plaintext.encode()).hexdigest()
reduced = reduce(md5_digest)

print("Plaintext: ", plaintext)
print("MD5sum:    ", md5_digest)
print("Reduced:   ", reduced)
