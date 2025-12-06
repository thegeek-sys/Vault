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
The encryption happens character per character. Mathematically, if we associate the numbers from 0 to 25 to the letters (A=0, B=1, …), we obtain the encrypted letter by summing the numeric value of the plaintext character and the numeric value of the letter of the key (everything module 26)

>[!example]
>![[Pasted image 20251205210129.png]]

### One-time pad
One-time pad is a vigenére cipher that uses a key as long as the ciphertext

>[!example]
>![[Pasted image 20251205210233.png]]

Thanks to the **Shannon theorem** we can say that this cipher is unbreakable. In fact to be perfect, in a cipher there must be at least as many keys as there are possible messages

#### Weaknesses of the one-time pad
In spite of their perfect security, one time pads have some weaknesses. In particular the kay has to be as long as the plaintext and the keys can never be reused (repeated use of one-time pads allowed the U.S. to break some of the communications of Soviet spies during the Cold War)

---
## Transposition ciphers
This kind of ciphers consist of changing the order of the letters in the message

>[!info] Those does not change the character frequency

### Rail fence
Given a message, arrange it in a zig-zag pattern and read the message by row, to decrypt split in two the message and read zig-zag

>[!example]
>![[Pasted image 20251205210905.png]]

### Permutation
Split it in blocks of length $m$ and rearrange each block with the same permutation (the key). To decrypt, apply the reverse permutation on the blocks of the ciphertext

>[!example]
>$(1, 2, 3, 4, 5)\to (3, 1, 2, 5, 4)$
>![[Pasted image 20251205211156.png]]

### Columnar transposition
Write the plain text up row by row with a fixed length (the key). To decrypt divide the message length for the key to find the number of columns and write the ciphertext by columns

>[!example]
>Key: row width $n=5$
>![[Pasted image 20251205211324.png]]

#### Keyed columnar transposition
Write the plaintext by row with a fixed row length, rearrange the columns according to a permutation, and write the text by columns

>[!example]
>![[Pasted image 20251205211655.png]]

---
## Modern cryptography
High redundancy of natural language often makes it possible to analyze the text using statistics and the key can be revealed, and this it particularly easy if the ciphertext bits of the plaintext is known.
For this reason it has been reached the conclusion that more complex codes are needed

The basic idea to modern cryptography comes from Claude Shannon (1949), who said that a ciphertext needs to have this two characteristics:
- **diffusion** → spread redundancy around the ciphertext
- **confusion** → makes encryption function as complex as possible making it difficult to derive the key analyzing the ciphertext

An encryption is computationally secure if:
- cost of breaking the cipher exceeds value of information
- time required to break cipher exceeds the useful lifetime of the information (usually is very difficult to estimate the amount of effort required to break)

### Feistel network
In 1973 Feistel proposed the concept of a **product cipher**. The idea was that the execution of two or more simple ciphers in sequence in such a way that the final result or product is cryptographically stringer than any of the component ciphers.
So Feister proposed the use of a cipher that alternates substitutions and permutations

![[Pasted image 20251206162324.png|300]]
In this case the plaintext is divided into 2 halves $L_{0}$ and $R_{0}$. $R_{0}$ then if combined and processed trough a function $F$ with a key from the round $K_{i+1}$. The result of this transformation then is summed (XOR) with the left part $L_{0}$ and the two resulting parts are then exchanged. This process is done $n$ times, and we will end up with the ciphertext

---
## Block ciphers
![[Pasted image 20251206163025.png]]

In a block cipher the plaintext of length $n$ is equally partitioned into a sequence of $m$ blocks $P[0],\dots,P[m-1]$. These partitions are called *blocks*

![[Pasted image 20251206163258.png]]

---
## Computers and cryptography
Modern codes tend to operate with messages ad binary data where every character in a message is encoded as a unique sequence of 0 and 1

In computers the substitution is often made with the XOR function

>[!example] One-time pad
>![[Pasted image 20251206163512.png]]

