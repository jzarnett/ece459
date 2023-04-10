from hashlib import md5
from sys import argv

def reduce(digest):
  reduced = ""
  for c in digest:
    if c.isdigit():
      reduced = reduced + c
      if len(reduced) == 7:
        return reduced

plaintext=argv[1]
md5_digest = md5(plaintext.encode()).hexdigest()
reduced = reduce(md5_digest)

print ("Plaintext: ", plaintext)
print ("MD5sum:    ", md5_digest)
print ("Reduced:   ", reduced)

