---
Class: "[[Cybersecurity]]"
Created: 2025-10-01
Related:
---
---
## Index
- [[#Symmetric encryption|Symmetric encryption]]
	- [[#Symmetric encryption#Attacking symmetric encryption|Attacking symmetric encryption]]
		- [[#Attacking symmetric encryption#Cryptoanalisys attacks|Cryptoanalisys attacks]]
		- [[#Attacking symmetric encryption#Brute-force attacks|Brute-force attacks]]
- [[#Most known symmetric encryption algorithms|Most known symmetric encryption algorithms]]
	- [[#Most known symmetric encryption algorithms#AES|AES]]
	- [[#Most known symmetric encryption algorithms#DES|DES]]
	- [[#Most known symmetric encryption algorithms#RC4|RC4]]
- [[#Practical security issues|Practical security issues]]
- [[#Block and stream ciphers|Block and stream ciphers]]
	- [[#Block and stream ciphers#Block cipher|Block cipher]]
	- [[#Block and stream ciphers#Stream cipher|Stream cipher]]
- [[#Message authentication|Message authentication]]
	- [[#Message authentication#Message authentication without confidentiality|Message authentication without confidentiality]]
	- [[#Message authentication#Cryptographic hash function|Cryptographic hash function]]
		- [[#Cryptographic hash function#Properties of a hash function aimed at authentication|Properties of a hash function aimed at authentication]]
		- [[#Cryptographic hash function#Security of hash functions|Security of hash functions]]
	- [[#Message authentication#Message Authentication Code|Message Authentication Code]]
		- [[#Message Authentication Code#MAC with one-way hash functions|MAC with one-way hash functions]]
- [[#Public-Key encryption structure|Public-Key encryption structure]]
	- [[#Public-Key encryption structure#Applications for public-key cryptosystems|Applications for public-key cryptosystems]]
		- [[#Applications for public-key cryptosystems#Requirements|Requirements]]
	- [[#Public-Key encryption structure#Asymmetric encryption algorithms|Asymmetric encryption algorithms]]
		- [[#Asymmetric encryption algorithms#RSA|RSA]]
		- [[#Asymmetric encryption algorithms#Diffie-Hellman key exchange algorithm|Diffie-Hellman key exchange algorithm]]
		- [[#Asymmetric encryption algorithms#Digital Signature Standard (DSS)|Digital Signature Standard (DSS)]]
		- [[#Asymmetric encryption algorithms#Elliptic curve cryptography (ECC)|Elliptic curve cryptography (ECC)]]
- [[#Digital signature|Digital signature]]
- [[#Public key certificate use|Public key certificate use]]
- [[#Digital envelope|Digital envelope]]
- [[#Random numbers|Random numbers]]
	- [[#Random numbers#Requirements|Requirements]]
	- [[#Random numbers#Random vs. Pseudorandom|Random vs. Pseudorandom]]
---
## Symmetric encryption
The universal technique for providing confidentiality for transmitted or stored data is **symmetric encryption**, also called conventional encryption or single-key encryption
1
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
## Message authentication
The message authentication is used to verify if a message has not been altered and that comes from the real legitimate sender

### Message authentication without confidentiality
Message encryption by itself does not provide a secure form of authentication, but we can solve it by combining authentication and confidentiality in a single algorithm

Typically message authentication is separate from message encryption

### Cryptographic hash function
![[Pasted image 20251009225137.png|320]]

The purpose of a hash function is to produce a “fingerprint” of a file, message, or other block of data. It generates a set of $k$ bits from a set of $L$ bits

The result of applying a hash function is called **hash value**, or message digest, or checksum

#### Properties of a hash function aimed at authentication
- can be applied to a block of data of any size
- produces a fixed length output
- $H(x)$ is relatively easy to compute for any given $x$
- one-way or pre-image resistance → computationally infeasible to find $x$ such that $H(x)=h$
- computationally infeasible to find $y\neq x$ such that $H(y)=H(x)$
- collision resistant → computationally infeasible to find any pair $(x,y)$ such that $H(x)=H(y)$

#### Security of hash functions
There are two approaches to attacking a secure hash function:
- exploit logical weakness in the algorithm
- strength of hash function depends solely on the length of the hash code produces by the algorithm

SHA is the most widely used hash algorithm

>[!tip] Additional secure hash function applications
>- hash of a password is stored by an operating system
>- store $H(F)$ for each file on a system and secure the hash values

### Message Authentication Code
The **Message Authentication Code** (*MAC*) is typically a part of the result of applying an encryption algorithm to a message then attached to the message and used by the receiver to assure authentication

![[Pasted image 20251009224654.png|450]]

#### MAC with one-way hash functions
Unlike the MAC, a hash function does not take a secret key as input, however it is possible to get MACs using hash functions

>[!example] Using symmetric encryption
>![[Pasted image 20251009225708.png|450]]

>[!example] Using public-key encryption
>![[Pasted image 20251009225753.png|450]]

>[!example] Using secret value
>![[Pasted image 20251009225829.png|450]]

---
## Public-Key encryption structure
Publicly proposed by Diffie and Hellman in 1976, the public-key encryption structure is based on mathematical functions and it’s **asymmetric**.
So it uses two separate  keys, private and public key, one used by the sender to encrypt and one used by the receiver to decrypt

>[!example] Encryption with public key
>![[Pasted image 20251009231028.png]]

>[!example] Encryption with private key
>![[Pasted image 20251009231105.png]]
>
>>[!error]
>>Using your private key to encrypt is not safe (anyone could the public keys and decrypt). It is used just to guarantee authenticity and integrity, **not for confidentiality**

In summary it is possible both to encrypt with the public key of the receiver (the receiver will then decrypt with his private key) and to encrypt with your private key (the receiver will then decrypt using the sender’s public key)


### Applications for public-key cryptosystems

| Algorithm      | Digital signature | Symmetric key distribution | Encryption of secret keys |
| -------------- | ----------------- | -------------------------- | ------------------------- |
| RSA            | yes               | yes                        | yes                       |
| Diffie-Hellman | no                | yes                        | no                        |
| DSS            | yes               | no                         | no                        |
| Elliptic Curve | yes               | yes                        | yes                       |

In board terms, we can classify the use of public-key cryptosystems into three categories:
- digital signature
- symmetric key distribution
- encryption of secret keys

#### Requirements
- computationally easy to create key pairs
- computationallly easy for sender knowing public key to encrypt messages
- computationally easy for receiver, knowing private key, to decrypt ciphertext
- useful if either key can be used for each role
- computationally infeasible for opponent to otherwise recover original message
- computationally infeasible for opponent to determine private key from public key

### Asymmetric encryption algorithms
#### RSA
The **RSA** (*Rivest, Shamir, Adleman*) was developed in 1977 and it’s the most widely accepted and implemented approach to public key encryption

The RSA is a block cipher in which the plaintext and ciphertext are integers between $0$ and $n-1$ for come $n$

#### Diffie-Hellman key exchange algorithm
This algorithm enables two parties to securely reach agreement about a shared secret for subsequent symmetric encryption of messages and is limited to the exchange of the keys

#### Digital Signature Standard (DSS)
The DSS provides only a digital signature function with SHA-1 and cannot be used for encryption or key exchange

#### Elliptic curve cryptography (ECC)
The ECC is as safe as the RSA, but with much smaller keys

---
## Digital signature

>[!quote] NIST FIPS PUB 186-4
>”The result of a cryptographic transformation of data that, when properly implemented, provides a mechanism for verifying origin authentication, data integrity and signatory non-repudiation”

Thus, a digital signature is a data-dependent bit pattern, generated by an agent as a function of a file, message, or other form of data block

FIPS 186-4 specifies the use of one of three digital signature algorithms:
- **Digital Signature Algorithm** (DSA)
- **RSA Digital Signature Algorithm**
- **Elliptic Curve Digital Signature Algorithm** (ECDSA)

>[!example] Digital signature process
>Bob signs a message and Alice verifies the signature
>![[Pasted image 20251010110124.png|400]]

---
## Public key certificate use
The certificate let you associate an entity to its public key and is guaranteed by a certification authority

![[Pasted image 20251010110404.png]]

---
## Digital envelope
In this case you use asymmetric encryption to encrypt the symmetric key. Then the message (encrypted with the symmetric key) and the encrypted symmetric key are sent to the receiver.

The receiver now has to decrypt the symmetric key using it’s private asymmetric key and then he can decrypt the message using the symmetric key

![[Pasted image 20251010110703.png|500]]

---
## Random numbers
In cryptography random numbers are essential. In fact they are used for the generation of:
- keys for public-key algorithms
- stream key for symmetric stream cipher
- symmetric key for use as a temporary session key or in creating a digital envelope
- handshaking to prevent replay attacks
- session key

### Requirements
Random numbers have very specific requirements to be really defined “random”:
- **randomness**
	- uniform distribution → frequency of occurrence of each of the numbers should be approximately the same
	- independence → no value in the sequence can be inferred from the others
- **unpredictability**
	- each number is statistically independent of other numbers in the sequence
	- opponent should not be able to predict future elements of the sequence on the basis of earlier element

### Random vs. Pseudorandom
Cryptographic applications typically make use of algorithmic techniques for random number generation, because algorithms are **deterministic** and therefore produce sequences of numbers that are not statistically random

**Pseudorandom numbers** are:
- sequences produced that satisfy statistical randomness tests
- likely to be predictable

**True random number generator** (TRNG):
- uses a nondeterministic source to produce randomness
- most operate by measuring unpredictable natural processes (eg. radiation, gas discharge, leaky capacitors)
- increasingly provided on modern processors

Pseudorandom numbers are predictable because they use a key to initialize the algorithm (same key produce same numbers), and for this reason is typically used a TRNG to generate the key for the initialization

