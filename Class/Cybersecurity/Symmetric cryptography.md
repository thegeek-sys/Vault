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
## Computers and cryptography
Modern codes tend to operate with messages ad binary data where every character in a message is encoded as a unique sequence of 0 and 1

In computers the substitution is often made with the XOR function

>[!example] One-time pad
>![[Pasted image 20251206163512.png]]

### Substitution boxes
The substitution in modern ciphers are not made by a simple characters swap, but they are defined from structures called **substitution boxes**, which has the objective of executing a non linear substitution to mask the relation between the ciphertext and the plaintext

>[!example]
>A number is split in 2 blocks: the first is the row the second the column
>
>![[Pasted image 20251206164105.png|500]]


---
## Block ciphers
![[Pasted image 20251206163025.png]]

In a block cipher the plaintext of length $n$ is equally partitioned into a sequence of $m$ blocks $P[0],\dots,P[m-1]$. These partitions are called *blocks*

![[Pasted image 20251206163258.png]]

### In practice
Nowadays exist many block ciphers:
- Data Encryption Standard (DES) → developed by IBM and adopted by NIST in 1977; consist of 64-bit blocks and 56-bit keys (small key space make exhaustive search feasible since late 90s)
- Triple DES (3DES) → nested application of DES with three different keys $KA$, $KB$, $KC$ so that the effective key length if $168$ bits, making exhaustive search attacks unfeasible (equivalent to DES when $KA=KB=KC$); $C=E_{KC}(D_{KB}(E_{KA}(P)))$, $P=D_{KA}(E_{KB}(D_{KC}(C)))$
- Advanced Encryption Standard (AES)

### Data Encryption Standard (DES)
This is the most widely used encryption scheme, and was adopted in 1977 by National Bureau of Standards (now NIST). The algorithm is referred to as the Data Encryption Algorithm (DEA)

It has minor variations from the Feistel network, but in 1999 it was considered from NIST no longer safe 

Although DES standard is public it was surrounded by considerable controversy. The primary points of contention focused on two areas:
- key length → there was debate over the choice of a 56-bit key, particularly when compared to other ciphers that used 128 bit keys
- classified criteria → the design criteria of the algorithm were classified (S-boxes may have backdoors)

But subsequent events and public analysis showed that the design was appropriate. Later research indicated that techniques like Differential Cryptoanalysis were less effective against DAS than initially feared

### Double DES
In 1992 it was shown that two DES encryption by  DES are not equivalend to a single encryption. In fact $E(K_{2}, E(K_{2}, M))$ is not equal to $E(K_{3},M)$ for any $K_{3}$

So multiple encipherment should be effective

#### Meet in the middle attacks
This attack proves that the security of Double DES is not as expected. In fact as key size is $56+56=112$ bits, the expected security is bruteforcing $2^{112}$ keys, but this was not the case

In fact the intermediate encryption is $E_{K_{1}}(\text{plain})=D_{K_{2}}(\text{cipher})$. So given a known pair of plaintext and ciphertext you can encrypt plain with $2^{56}$ keys and decrypt the cipher with $2^{56}$ keys and compare the two results to find the matching intermediate text

### Triple DES
3DES is nested application of DES that typically uses three different keys $(K_{1},K_{2},K_{3})$ and applies sequential encryption (with $K_{1}$), decryption (with $K_{2}$), and encryption (with $K_{3}$). Let's see analyze this three cases:
- $K_{1}=K_{2}=K_{3}$ → DES
- $K_{1}=K_{3},K_{2}$ → $2^{112}$
- $K_{1},K_{2},K_{3}$ → $2^{168}$

3DES with two keys is a relatively popular alternative to DES

### Advances Encryption Standard (AES)
In 1997, the U.S. National Institute for Standards and Technology (NIST) put out a public call for a replacement to DES. The chosen winner was Rijndael algorithm which standardized the AES

AES is a block cipher that operates on 128 bit blocks and can use keys that are 128, 192, or 256 bits long

![[Pasted image 20251206172032.png|500]]

#### AES round structure
128-bit AES uses 10 rounds and each round is an invertible transformation the initialization is $X_{0}=P\oplus K$ and the cipher text $C$ is $X_{10}$

![[Pasted image 20251206173432.png|250]]

Each round is built from four basic steps:
- *SubBytes* → S-box substitution step
- *ShiftRows* → permutation step
- *MixColumns* → matrix multiplication step
- *AddRoundKey* → XOR step with a round key derived from the 128-bit encryption key

### Bruteforcing modern block ciphers
| Dimensione Chiave (bits)    | Cifrario       | Numero di Chiavi Alternative         | Tempo Necessario (a 109 Decifrazioni/s)                         | Tempo Necessario (a 1013 Decifrazioni/s) |
| --------------------------- | -------------- | ------------------------------------ | --------------------------------------------------------------- | ---------------------------------------- |
| 56                          | DES            | $2^{56} \approx 7.2 \times 10^{16}$  | $2^{55} \text{ ns} = 1.125 \text{ years}$                       | 1 hour                                   |
| 128                         | AES            | $2^{128} \approx 3.4 \times 10^{38}$ | $2^{127} \text{ ns} = 5.3 \times 10^{21} \text{ years}$         | $5.3 \times 10^{17} \text{ years}$       |
| 168                         | Triple DES     | $2^{168} \approx 3.7 \times 10^{50}$ | $2^{167} \text{ ns} = 5.8 \times 10^{33} \text{ years}$         | $5.8 \times 10^{29} \text{ years}$       |
| 192                         | AES            | $2^{192} \approx 6.3 \times 10^{57}$ | $2^{191} \text{ ns} = 9.8 \times 10^{40} \text{ years}$         | $9.8 \times 10^{36} \text{ years}$       |
| 256                         | AES            | $2^{256} \approx 1.2 \times 10^{77}$ | $2^{255} \text{ ns} = 1.8 \times 10^{60} \text{ years}$         | $1.8 \times 10^{56} \text{ years}$       |
| 26 caratteri (permutazione) | Monoalfabetico | $26! \approx 4 \times 10^{26}$       | $2 \times 10^{26} \text{ ns} = 6.3 \times 10^{6} \text{ years}$ | $6.3 \times 10^{6} \text{ years}$        |

---
## Stream ciphers
In stream ciphers the encryption scheme/key can change for each symbol of the plaintext.
Given a plaintext $m_{1},m_{2},\dots$ and a keystream $e_{1},e_{2},\dots$ produces the ciphertext $c_{1},c_{2},\dots$ where $c_{i}=E(e_{i},m_{i})$ and $m_{i}=D(e_{i},c_{i})$. The encryption and decryption is usually done trough the xor

>[!hint]
>In some sense, stream ciphers are block ciphers with block size of length one

Those are useful when plaintext needs to be processed one symbol at a time or the message is short (short message with block ciphers needs padding)

### Block and stream ciphers
A block cipher operates on a fixed length of contiguous characters at a time where each block is considered independent, requiring the use of the key repeatedly for each block of data. Block ciphers are standardized and more widely available

A stream cipher treats the message to be encrypted as one continuous stream of characters (one-time pad can be considered a stream cipher where the key length is equal to the message length)

![[Pasted image 20251206174619.png|600]]

### Characteristics of stream cipher
- it should have long periods without repetition (e.g. RC4 period is estimated $10^{100}$)
- it needs to depend on large enough key
- possibly each keystream bit should depend on most or all of the cryptovariable bits
- statistically upredictable
- keystream should be statistically unbiased
- advantages → speed of transformation, no error propagation
- disadvantages → low diffusion, subject to malicious insertion and modification

### RC4
RC4 is a proprietary stream cipher owned by RSA and designed by Ron Rivest.
It can have variable keys size (from 1 to 256 bytes) and have byte-oriented operations.

It is widely used on web and key forms random permutation of all 8-bit values; permutation used to scramble input info are processed one byte at a time. It is also secure against known attacks

Simple code structure:
```
/* Initialization */
for i = 0 to 255 do
	S[i] = i;
	T[i] = K[i mod keylen];

/* Initial Permutation of S */
j = 0;
for i = 0 to 255 do
	j = (j + S[i] + T[i]) mod 256;
	Swap (S[i], S[j]);

/* Stream Generation */
i, j = 0;
while (true)
	i = (i + 1) mod 256;
	j = (j + S[i]) mod 256;
	Swap (S[i], S[j]);
	t = (S[i] + S[j]) mod 256;
	k = S[t];
	Output k;
```

![[Pasted image 20251206180105.png]]