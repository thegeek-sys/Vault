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
>- what is the size of the chipertext, relative to the plaintext?
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
It consists trying all the possible keys $K$ and determine if $D_{k}(C)$ is a likely plaintext but requires some knowledge of the st