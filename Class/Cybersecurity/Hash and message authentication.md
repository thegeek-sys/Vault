---
Class: "[[Cybersecurity]]"
Related:
---
---
## Cryptographic hash function
The purpose of a hash function is to produce a “fingerprint” of a file, message, or other block of data. It generates a set of $k$ bits from a set of $L$ ($\geq k$) bits (in general not necessarily injective)

The result of applying a hash function is called **hash value** or message digest, or checksum

![[Pasted image 20251213160253.png|500]]

### Properties
The properties for a hash function aimed at authentication are:
- it can be applied to a block of data of any size
- produces a fixed-length output
- $H(x)$ is relatively easy to compute for any given $x$
- it is one-way or pre-image resistant → is computationally infeasible to find $x$ such that $H(x)=h$
- is computationally infeasible to find $y\neq x$ such that $H(x)=H(y)$ (*second preimage resistant*)
- it is collision resistant or have a strong collision resistance → is computationally infeasible to find any pair $(x,y)$ such that $H(x)=H(y)$

### Security of hash functions
There are two approaches to attacking a secure hash function:
- exploit logical weaknesses in the algorithm (e.g. cryptoanalysis)
- brute forcing

**SHA** (*Secure Hash Algorithm*) is the family of algorithms most widely spread to guarantee this security

These hash function have two other uses:
- store encrypted passwords
- check the file integrity by storing $H(F)$ for each file on a system

>[!example] Simple hash functions with bit-by-bit exclusive-OR
>One of the simplest hash functions is the bit-by-bit exclusive-OR (XOR) of every block
>![[Pasted image 20251213162607.png]]
>
>$$C_{i}=b_{i1}\oplus b_{i 2}\oplus\dots \oplus b_{im}$$
>
>where:
>- $C_{i}$ → $i$-th bit of the hash code
>- $1\dots i\dots n,m$ → number of $n$-bit blocks in the input
>- $b_{ij}$ → $i$-th bit in $j$-th block

### Secure Hash Algorithm (SHA)
SHA was originally developed by NIST and published as FIPS 180 in 1993 but was revised in 1995 as SHA-1 (produces 160-bit hash values). Then NIST issued revised FIPS 180-2 in 2002 by adding 3 additional versions of SHA (SHA-256, SHA-384, SHA-512) with 256/384/512-bit hash values.

The most recent version is FIPS 180-4 which added two variants of SHA-512 with 224-bit and 256-bit hash sizes

![[Pasted image 20251213162945.png]]

>[!info]
>Security refers to the fact that a birthday attack on a message digest of size $n$ produces a collision with a work factor of approximately $2^{n/2}$
>
>>[!hint] Birthday paradox
>>Is much harder to find someone in a room full of people with the same birth date as you but it’s way easier to find two person with the same birth date. So an attacker to find any collision has to try just $2^{n/2}$ different hashes, not $2^n$

#### Message digest generation with SHA-512
![[Pasted image 20251213163636.png]]

This consist of three phases:
- message preparation
	- if the message length is less than a multiple of 1024 bits it is filled with a 1 followed by zeros and at the end of the message $L$, containing the actual length of the message, is attached. The message is then divided into 1024 bit blocks
- chain process
	- as the first block ($M_{1}$) has no previous hash it is used a 512 bit initial vector. At this point for each block it gives the result of the previous block ($H_{i-1}$) and the block ($M_{i}$) to the hash function ($F$). The obtained results are then added word-by-word mod $2^{64}$

#### SHA-512 single 1024-bit processing
Let’s now analyze what happens in each $F$

![[Pasted image 20251213165247.png|400]]

Each round $t$ makes use of a 64-bit value $W_{t}$, derived from the current 1024-bit block being processed ($M_{i}$) and of an additive constant $K_{t}$

The operations performed during a round consist of circular shifts, and primitive boolean functions based on AND, OR, NOT and XOR

#### SHA-3
SHA-2 shares the same structure and mathematica operations as its predecessor and causes concern (SHA-1 and MD5 that had been exploited), even tho it is still considered scure. For this reason in 2007 announced a competition to produce SHA-3. These were the requirements

>[!info] Requirements
>- must support hash value lengths of 224, 256, 384 and 512 bits
>- the algorithm must process small blocks at a time instead of requiring the entire message to be buffed in memory before processing it

---
## Message authentication
Let’s see message authentication properties:
- protects against active attacks
- verifies if received message is authentic → contents have not been altered, from authentic source, timely and in correct sequence
- can use conventional encryption → only sender and received share a key

### Message authentication without confidentiality
Message encryption by itself does not provide a secure form of authentication but we can combine authentication and confidentiality in a single algorithm (encryption + authentication tag)

Typically message authentication is separate from message encryption.

>[!example] Examples for message authentication without confidentiality
>- applications in which the same message s broadcast to a number of destinations
>- an exchange in which one side has a heavy load and cannot afford the time to decrypt all incoming messages
>- authentication of a computer program in plaintext

### Message authentication code (MAC)
![[Cryptographic concepts#Message Authentication Code]]

### HMAC
The HMAC (Hash-based Message Authentication Code) is the most used way to obtain message authentication using hash functions

The key point is the interest in developing a MAC derived from a cryptographic hash function. This approach is favored for two main reasons:
- speed and efficiency: → cryptographic hash functions generally execute faster than symmetric encryption algorithms
- code availability → the library code for hash functions like SHA is widely available across operating systems

Tho SHA-1 (and other hash functions) were not designed for use as a MAC because they do not rely on a secret key (HMAC solves authentication by securely incorporating a shared secret key (K) into the hash calculation process)

It was issued as RFC2014 and it has been chosen as the mandatory-to-implement MAC for IP security but it is used also in other Internet protocols such as Transport Layer Security (TLS) and Secure Electronic Transaction (SET)

>[!info] HMAC design objectives
>- to use, without modifications, available hash functions
>- to allow for easy replaceability of the embedded hash function in case faster or more secure hash functions are found or required
>- to preserve the original performance of the hash function without incurring a significant degradation
>- to use and handle keys in a simple way
>- to have a well-understood cryptographic analysis of the strength of the authentication mechanism based on reasonable assumptions on the embedded hash function

#### Structure
HMAC can be expressed as follows:
$$
\text{HMAC}(K,M)=H[(K^+ \oplus \text{opad}) \mid\mid H[(K^+ \oplus \text{ipad}) \mid \mid M]]
$$

where $\text{ipad}$ and $\text{opad}$ are fixed

HMAC should execute in approximately the same time as the embedded hash function for long messages, in fact the overhead of the double hash is negligible compared to the time needed to process the entire message in blocks

![[Pasted image 20251214085337.png|400]]

In essence the HMAC is a double hashing process that uses the secret key to “pad” the message and conceal it within two nested hash calculations, thereby ensuring that only the key holder can create or verify the authentication code

### Security
Security of HMAC depends on the cryptographic strength of the underlying hash function. The appeal of HMAC is that its designers have been able to prove an exact relationship  between the strength of the embedded hash function and the strength of HMAC

For a given level of effort on messages generated by a legitimate user and seen by the attacker, the probability of a successful attack on HMAC is equivalent to one of the following attacks on the embedded hash function:
- the attacker is able to compute an output of the compression function even with an IV that is random, secret, and unknown to the attacker
- the attacker finds collisions in the hash function even when the IV is random and secret
