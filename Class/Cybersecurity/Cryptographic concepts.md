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
The **AES** (*Advances Encryption Standard*), also known as **Rikndael**, is the most popular and widely used encryption algorithm in the modern IT industry

It uses $128$ block cipher with $128$, $192$ (between 128 and 256) or $256$ bit secret keys

### DES
The **DES** (*Data Encryption Standard*) is now considered insecure due to the shortness of the key but was the most widely used encryption scheme before AES

In fact it uses $64$ bit block cipher and $56$ bit secret key to produce 64 bit ciphertext block

>[!error]
>Some people think that was chosen $56$ bit secret key because at that time just the NSA had enough powerful computers able to decrypt that kind of chipertext

Due to its lack of safeness nowadays, it was introduced **3DES** which consists of cascading encrypting the same plaintext three times with three different keys

### RC4
The **RC4**, also known as *ARC4* or *ARCFOUR*, is now considered insecure

It used a *stream cipher* with $40-2048$ bits secret keys

