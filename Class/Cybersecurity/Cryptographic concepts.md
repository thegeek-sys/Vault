---
Class: "[[Cybersecurity]]"
Created: 2025-10-01
Related:
---
---
## Symmetric encryption
The universal technique for providing confidentiality for transmitted or stored data is **symmetric encryption**, also called conventional encryption or single-key encryption

It has two requirements for secure use:
- needs a strong encryption algorithm (AES standard wants a key 128 bit long)
- sender and receiver must have obtained copies of the secret key in a secure fashion and must keep the key secure

![[Pasted image 20251001224639.png|center|600]]

### Attacking symmetric encryption
Even if it is safer than asymmetric encryption (also regarding quantum computing) it can still suffer from some kind of attacks

#### Cryptoanalisys attacks
This kind of attack relies on the nature of the algorithm (it’s deterministic). By using some knowledge of the general characteristics of the plain text like or some sample plaintext-ciphertext pairs it can exploit the characteristics of the algorithm to deduce a specific plaintext or the key being used (even if the main target is the key, not the text)

>[!hint]
>It’s mainly used to reduce the dictionary for a possible brute-force attacks, but nowadays it is outdated due to the new standards of encryption

>[!example]
>Is you encrypt two times the same block with the same key, the ciphertext will be the same

#### Brute-force attacks
This kind of attack just tries all possible keys on some ciphertext until an intelligible transaltion into plaintext is obtain

>[!info]
>On average half of the possible keys must be tried to achieve success

---
## Most known symmetric encryption algorithms
![[Pasted image 20251001231041.png]]

### AES
The **AES** (*Advances Encryption Standard*), also known as **Rikndael**, is the most popular and widely used encryption algorithm in the modern IT industry. It was published by the NIST in 1997, and should have a security equal to or better than 3DES but it is much faster in terms of efficiency

It uses $128$ block cipher with $128$, $192$ (between 128 and 256) or $256$ bit secret keys

### DES
The **DES** (*Data Encryption Standard*) is now considered insecure due to the shortness of the key but was the most widely used encryption scheme before AES

In fact it uses $64$ bit block cipher and $56$ bit secret key to produce 64 bit ciphertext block

>[!error]
>Some people think that was chosen $56$ bit secret key because at that time just the NSA had enough powerful computers able to decrypt that kind of chipertext

Due to its lack of safeness nowadays, it was introduced **3DES** which consists of cascading encrypting the same plaintext three times with three different keys. It was first standardized by ANSI in 1985 as it solves the security problem that the original DES had but it is sluggish in software and uses a 64 bit block size

### RC4
The **RC4**, also known as *ARC4* or *ARCFOUR*, is now considered insecure

It used a *stream cipher* with $40-2048$ bits secret keys

---
## Practical security issues
Typically symmetric encryption is applied to a unit of data larger than a single 64 bit or 128 bit block

**Electronic codebook** (*ECB*) mode is the simplest approach to multiple-block encryption, but each block of plaintext is encrypted using the same key so that cryptoanalyst may be able to exploit regularities in the plaintext

For this reason were developed alternative techniques to increase the security of symmetri block encryption for large sequences, overcoming the weaknesses of ECB

---
## Block and stream ciphers
![[Pasted image 20251009223314.png|center|500]]

### Block cipher
A block cipher processes the input one block of elements at a time and produces an output block for each input block

It’s the widely used and can reuse keys

### Stream cipher
A stream cipher processes the input elements continuously and produces output one element at a time (it encrypts plaintext one byte at a time)

The primary advantage is that they are almost always faster and use far less code. A pseudorandom stream is one that is unpredictable without knowledge of the input key

---
## Cryptographic hash function
![[Pasted image 20251009225137.png|320]]

The purpose of a hash function is to produce a “fingerprint” of a file, message, or other block of data. It generates a set of $k$ bits from a set of $L$ bits

The result of applying a hash function is called **hash value**, or message digest, or checksum

## Message authentication
The message authentication is used to verify if a message has not been altered and that comes from the real legitimate sender

### Message authentication without confidentiality
Message encryption by itself does not provide a secure form of authentication, but we can solve it by combining authentication and confidentiality in a single algorithm 

Typically message authentication is separate from message encryption

### Message Authentication Code
The **Message Authentication Code** (*MAC*) is typically a part of the result of applying an encryption algorithm to a message then attached to the message and used by the receiver to assure authentication

![[Pasted image 20251009224654.png|450]]

#### MAC with one-way hash functions
Unlike the MAC, a hash function does not take a secret key as input, however it is possible to get MACs using hash functions