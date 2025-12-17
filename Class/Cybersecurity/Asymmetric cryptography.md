---
Class: "[[Cybersecurity]]"
Related:
---
---
## Introduction
Asymmetric encryption was publicly proposed by Diffie and Hellman in 1976 and it is based on mathematical functions
It is called asymmetric as it used two separate keys, a public and a provate key (public key is made public for others to use), but some form of protocol is needed for distribution

### Requirements for public-key cryptosystems
- computationally easy to create key pairs
- computationally easy for sender knowing public key to encrypt messages
- computationally easy for receiver knowing private key to decrypt ciphertext
- useful if either key can be used for each role
- computationally infeasible for opponent to otherwise recover original message
- computationally infeasible for opponent to determine private key from public key

### Applications
| Algorithm      | Digital signature | Symmetric key distribution | Encryption of secret keys |
| -------------- | ----------------- | -------------------------- | ------------------------- |
| RSA            | yes               | yes                        | yes                       |
| Diffie-Hellman | no                | yes                        | no                        |
| DSS            | yes               | no                         | no                        |
| Elliptic Curve | yes               | yes                        | yes                       |

In broad terms, we can classify the use of public-key cryptosystems into three categories: digital signature, symmetric key distribution, and encryption of secret keys.

### Advantages
- there is no need to communicate private key → related public key is widely distributed (not kept secret)
- a sender who private-key encrypts the message or any part thereof can be authenticated → no one else is supposed to have the sender’s private key
- external parties can confidentially communicate with an owner of the key pair by sending a message encrypted using the owner’s public key
- a brute force on a message encrypted using PKC is time consuming and is nearly impossible

### Limitations
- the use of PKC takes a significant amount of processing power (it is **computationally intensive**) → therefore, it negatively affects efficiency of communication
- generally it is used **selectively** → an entire message may not be encrypted using PKC
- a private-key encrypted message can be decrypted by anyone (but it can turn as advantage, digital signature)
- published keys may be altered by someone → additional measures to ensure that a valid public key of the owner is obtained before its use (certificates)

---
## Encryption
### Encryption with public key
Each user generates a pair of keys to be used for the encryption and decryption of messages and places one of them in a public register or other accessible file (the public key) while the companion key is kept private.

![[Pasted image 20251214092317.png]]

Each user maintains a collection of public keys obtained from others, so if Bob wishes to send a private message to Alice, Bob encrypts the message using Alice’s public key. When Alice receives the message, she decrypts it using her private key (no other recipient can decrypt the message because only Alice knows Alice’s private key)

### Encryption with private key
A user encrypts data using his own private key. In this way anyone who knows the corresponding public key will then be able to decrypt the message

![[Pasted image 20251214092511.png]]

In this way is only guaranteed authentication, not confidentiality

---
## Asymmetric encryption algorithms
![[Cryptographic concepts#Asymmetric encryption algorithms]]

---
## Digital signature
A signature testifies/acknowledges some content; the signer links/binds himself to the content. So a digital signature is a way of electronically binding oneself to the content of a message or a document.
The way of doing this is by encrypting message digest (or the message) using one’s private key

![[Pasted image 20251214100018.png]]

>[!example]
>![[Pasted image 20251214100044.png|400]]

>[!example] Cryptography in today life
>Symmetric and asymmetric cryptography are used together. Asymmetric encryption is used to exchange a key that is used in symmetric cryptography , os that only few encryptions are done using PKC (RSA) while all the traffic is encrypted with SC (AES)
>
>![[Pasted image 20251214100849.png|400]]
>
>>[!info] Digital envelope
>>In this case you use asymmetric encryption to encrypt the symmetric key. Then the message (encrypted with the symmetric key) and the encrypted symmetric key are sent to the receiver.
>>
>>The receiver now has to decrypt the symmetric key using it’s private asymmetric key and then he can decrypt the message using the symmetric key
>>
>>![[Pasted image 20251010110703.png|500]]
>
>>[!question] Can we trust a public key?
>>The diagram illustrates a **Man-in-the-Middle (MITM)** attack . When the user requests the public key, an attacker intercepts the communication and substitutes the legitimate key with their own false public key. The unsuspecting user then encrypts sensitive data (like payment details) with the attacker's key. The attacker intercepts the data, decrypts it, reads the private information, re-encrypts the original message using the _actual_ public key of the store, and forwards it.
>>
>>The consequence is that the legitimate sender (the store) receives the message and believes the communication was secure, while the attacker has successfully eavesdropped on the confidential transaction. This highlights that Public-Key Cryptography alone is not enough; a mechanism, such as **Digital Certificates** issued by a **Certificate Authority (CA)**, is required to authenticate the ownership of public keys.
>>
>>![[Pasted image 20251214101318.png]]

---
## Digital certificates
A digital certificate is a document that certifies the relation between a public key and its owner though a digital signature, but to verify a digital signature we need another public key. For this reason we need a globally trusted public key

Trusted public keys are stored in certificates of **Certification Authorities** (*CA*)

![[Pasted image 20251217152714.png]]