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