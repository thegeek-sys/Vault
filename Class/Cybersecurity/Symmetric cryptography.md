---
Class: "[[Cybersecurity]]"
Related:
---
---
## Cryptography basic concepts
Cryptography is the field that offers techniques and methods of managing secrets. The primary purpose of cryptography is to alter a message so that only the intended recipients can alter it back and thereby read the original message

Purposes of cryptography:
- preserve confidentiality
- authenticate senders and receivers of messages
- facilitate message integrity
- ensure that the sender will not be able to deny transfer of message (non-repudiation)

>[!example] Symmetric cryptosystem
>Alice wants to send a message (plaintext $P$) to Bob. The communication channel is insecure and can be eavesdropped.
>
>If Alice and Bob have previously agreed on a symmetric encryption scheme and a secret key $K$, the message can be sent encrypted (ciphertext $C$)
>
>Issues:
>- what is a good symmetric encryption scheme?
>- what is the complexity of encrypting/decrypting?
>- what is the size of the ciphertext, relative to the plaintext?
>
> ![[Pasted image 20251205191654.png]]

### Definitions
- secret key → $K$
- encryption function → $E_{k}(P)$
- decryption function → $D_{k}(C)$
- plaintext length typically the same ciphertext length
- encryption and decryption are permutation functions (bijections) on the set of all n-bit arrays
- efficiency → functions $E_{x}$ and $D_{k}$ should have efficient algorithms
- consistency → decrypting the chipertext yields the plaintext $D_{k}(E_{k}(P))=P$

---
## Attacks to cryptography
### Brute force attack 
It consists trying all the possible keys $K$ and determine if $D_{k}(C)$ is a likely plaintext but requires some knowledge of the structure of the plaintext (e.g. PDF file or email message)

Key should be a sufficiently long random value to make exhaustive search attacks unfeasible

![[Pasted image 20251205200252.png|400]]

### Cryptoanalysis
Cryptoanalysis is the practice of breaking encrypted messages or codes to gain access to the original information without knowing the secret key but just some informations. The attacker may have:
- a collection of ciphertexts (*ciphertext only attack*)
- a collection of plaintext/ciphertext pairs  (*known plaintext attack*)
- a collection of plaintext/ciphertext pairs for plaintext selected by the attacker (*chosen plaintext attack*)
- a collection of plaintext/ciphertext pairs for ciphertexts selected by the attacker (*chosen ciphertext attack*)

---
## Symmetric key cryptography
It makes use of number of classical encryption techniques, mainly:
- substitution → each character of the plaintext is replaced by another character of the same or different alphabet (e.g. caesar cipher)
- transposition → just the order of the character in the text is changed, the value remains the same

Those operations are repeated multiple times

---
## Caesar cipher
The caesar cipher is a simple substitution cipher where each character is replaced in plaintext with the character 3 positions forward in the alphabet. If the end of the alphabet is reached, it starts over in the alphabet

>[!info]
>We can change 3 with any other number (it is the key)

![[Pasted image 20251205202911.png|400]]

### Weakness and improvement
With cyclic permutation, it is easy to find the key as there are only $N$ possibilities to try, where $N$ is the number of characters in the alphabet

An improvement could be random ptermutation of the alphabet and just then applying the substitution

>[!example]
>Alphabet → `ABCDEFGHIJKLMNOPQRSTUVWXYZ`
>Cipher code → `KEPALMUHDRVBXYSGNIZFOWTJQC`

As the caesar cipher is a single alphabet code, all the text is encoded with the same scheme of the alphabet

---
## Encrypting natural languages
English text typically is represented with 8-bit ASCII encoding, so a message with $t$ characters corresponds to an n-bit array with $n=8t$

Due to redundancy, the English plaintext (or in any other natural language), they are just a small subset of all the possible arrays of n-bit. This redundancy of of words or groups of letter makes those kind of ciphers vulnerable to **frequency analysis**

### Cryptoanalysis: frequency analysis
With frequency analysis, single alphabets substitution characters can be analyszed by calculating the frequencies of characters ina ciphertext and comparing the frequencies of characters in a ciphertext, and comparing the frequencies of characters in typical text of the same language


Frequency analysis can also be used on groups of charactes to get better results. For example considering the distribution of two character pairs (2-grams) in generic English text

---
## Poly-alphabetic ciphers
With the random permutation and a single alphabet, it is still relatively easy to find the key. Then we need something stronger, like a poly-alphabetic substitution cipher

In poly-alphabetic ciphers words are used as keys, where each character determines the displacement of the cipher alphabet, which will be applied to the character of the plaintext

Because of this, the same character in the plaintext may be represented by a different designated character

### More sophisticated substitutions
Crypotianalysis of ciphertextusing a poly-alphabetic cipher is therefore difficult (but not impossible)

>[!example] Cyclic permutation
>The key “FT” means to displace by 5 for characters in odd position in the original alphabet, and to displace by 19 for characters in even position
>
>![[Pasted image 20251205205330.png]]

### Vigenére code
The 

