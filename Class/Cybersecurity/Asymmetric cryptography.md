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

### Certification authority
A certification authority is an organization that issues digital certificates. The CA performs many tasks:
- receive application for keys
- verify applicant’s identity, conduct due diligence appropriate to the trust level, and issue key pairs
- store public keys and protect them from unauthorized modification
- keep a register of valid keys
- revoke and delete keys that are invalid or expired and maintain a certificate revocation list (CRL)

Certificates of CAs are stored in any computer that want to use internet securely.
A user can present his/her public key to the authority in a secure manner to obtain a certificate, so that he can then publish it (or send it to others) to make possible for everyone to obtain and verify the certificate 

>[!question] How does HTTPS works?
>![[Pasted image 20251217155056.png]]

### Public Key Infrastructure
![[Pasted image 20251217153655.png|300]]

Certification authorities are organized in a hierarchy, called  **Public Key Infrastructure** (*PKI*), so to verify a certificate, one needs to verify all the signatures up to the top of the hierarchy

>[!info] Architectural model
>![[Pasted image 20251217154551.png|400]]

#### X.509
This standard is specified in RFC 5280 and is the most widely accepted format for public-key certificates, which are used in most network security applications, including:
- IP security (IPSEC)
- secure socket layer (SSL)
- secure electronic transactions (SET)
- S/MIME
- eBusiness applications

>[!quote] Certificate definition
>A public key with the identity of the key’s owner signed by a trusted third party
>Typically the third party is a CA that is trusted by the user community (such as a government agency, telecommunications company, financial institution, or other trusted peak organization)

>[!info] X.509 certificate
>![[Pasted image 20251217154507.png]]
>
>>[!example] Variants
>>A number of specialized variants also exist, distinguished by particular element values or the presence of certain extensions:
>>- conventional (long-lived) certificates → CA and “end user” certificates, typically issued for validity periods of months to years
>>- short-lived certificates → used to provide authentication for applications such as grid computing, while avoiding some of the overheads and limitations of conventional certificates; they have validity periods of hours to days, which limits the period of misuse if compromised because they are usually not issued by recognized CA’s there are issued with verifying them outside their issuing organization
>>- other → proxy certificates, attribute certificates

---
## RSA
RSA algorithm was created by Rivest, Shamir, and Adelman in 1977 and it’s based on the notion that a product of two large prime numbers cannot be easily factored to determine the two prime numbers

>[!hint]
>Although a public key is related to private key, it is nearly impossible to calculate the private key using the knowledge of its related public key
>

RSA is the best known and widely used public key algorithm and uses exponentiation of integers modulo a large number

$$
\text{encrypt: } C=M^e \text{ mod } n
$$
$$
\text{decrypt: } M=C^d \text{ mod } n=(M^e)^d \text{ mod }n=M
$$
where both sender and receiver know the values of $n$ and $e$, but only receiver knows the value of $d$

The keys are:
- public key → $PU=\{e,n\}$
- private key → $PR=\{d,n\}$

>[!info] RSA principles
>- consider data blocks as large numbers, for example $2048$ bit long number ($\sim 617$ decimal long number)
>- uses the modular arithmetic (residual), for example $73=70+3=14*5+3\to 73\text{ mod }5=3$
>- encryption and decryption are based on the concept of modular inverses, for example $X$ is the inverse of $Y$ modulo $Z$ if $X\cdot Y=1\text{ mod }Z$ then $(m^X)^Y=m^1=m$

### Keys generations
First of all two robust prime numbers $p$ and $q$ are chosen, then $n$ that composes both public and private keys is calculated as $n=p\cdot q$ → $n$ 

At this point is Euler’s Totient Functions $\phi(n)$ is applied (the number of positive numbers less than $n$ that are prime to $n$) as it follows
$$
\phi(p\cdot q)=(p-1)(q-1)
$$

To choose the exponent $e$ for the public key this rules are followed:
- $1<e<\phi(n)$
- $e$ has to be coprime to $\phi(n)$ (the greatest common divisor between $e$ and $\phi (n)$ has to be $1$)

The exponent $d$ for the private key is the modular inverse of $e$ modulo $\phi(n)$ so that $d\cdot e\equiv 1 \text{ mod } \phi(n)$

>[!example]
>Setup:
>- $n=187$ → $p=11$ and $q=17$
>- $\phi (n)=10\cdot 16=160$
>- $e=7$ → the greatest common divisor between $160$ and $7$ is $1$
>- $d=23$ → $23\cdot 7 \text{ mod } 160=161\text{ mod }160=1$
>
>Keys:
>- $PU=(7, 187)$
>- $PR=(23,187)$
>
>Encryption ($M=88$):
>- $C=88^7\text{ mod }187=11$
>
>Decryption ($C=11$):
>- $M=11^{23}\text{ mod }187=88$
>
>![[Pasted image 20251219085939.png]]

### Security of RSA
- brute force → involves trying all possible private keys
- mathematical attacks → there are several approaches, all equivalent in effort to factoring the product of two primes
- timing attacks → these depend on the running time of the decryption algorithm (trying to discover a private key by observing how long it takes to perform cryptographic operations)
- chosen ciphertext attacks → this type of attack exploits properties of the RSA algorithm

#### Timing attacks
Paul Kocher, a cryptographic consultant, demonstrated that a snooper can determine a private key by keeping track of how long a computer takes to decipher messages

Timing attacks are applicable not just to RSA, but also to other public-key cryptography systems. This attack is alarming for two reasons:
- it comes from a completely unexpected
- it is a ciphertext-only attack

>[!info] Countermeasures
>- constant exponentiation time
>	- ensure that all exponentiations take the same amount of time before returning a result (simple fix but degrades performance)
>- random delay
>	- better performance could be achieved by adding a random delay to the exponentiation algorithm to confuse the timing attacks
>	- if defenders do not add enough noise, attackers could still success by collecting additional measurements to compensate for the random delays
>- blinding
>	- multiply the ciphertext by a random number before performing exponentiation
>	- this process prevents the attacker from knowing what ciphertext bits are being processed inside the computer and therefore prevents the bit-by-bit analysis essential to the timing attack

---
## Diffie-Hellman key exchange
The Diffie-Hellman algorithm is the first published public-key algorithm by Diffie and Hellman in 1976 along with the exposition of public key concepts

It is currently used in a number of commercial products as a practical method to exchange a secret key securely that can be used for subsequent encryption of messages

The security of this algorithm relies on the difficulty of computing discere logarithms

>[!info] Key exchange protocol
>![[Pasted image 20251219100012.png]]

>[!example] Diffie-Hellman example
>Key exchange is base on the use of the prime number $q=353$ and a primitive root of $353$, in this case $\alpha=3$
>
>>[!info] Primitive root
>>A number $a$ is a **primitive root** of a prime number $p$ if the powers of a modulo $p$ generate every number from $1$ to $p−1$. In other words, a is a generator of the multiplicative group of integers modulo $p$.
>>>[!example] Example with $p=7$
>>>Let’s test if $3$ is a primitive root of $7$:
>>>- $3^1=3\equiv3(\text{mod }7)$
>>>- $3^2=9\equiv2(\text{mod }7)$
>>>- $3^3=27\equiv6(\text{mod }7)$
>>>- $3^4=81\equiv4(\text{mod }7)$
>>>- $3^5=243\equiv5(\text{mod }7)$
>>>- $3^6=729\equiv1(\text{mod }7)$
>
>$A$ and $B$ select secret keys $X_{A}=97$ and $X_{B}=233$, respectively
>- $A$ computes $Y_{A}=3^{97}\text{ mod 353}=40$
>- $B$ computes $Y_{B}=3^{253}\text{ mod 353}=248$
>
>After the exchange:
>- $A$ computes $K=(Y_{B})^{X_{A}} \text{ mod } 353=248^{97}\text{ mod }353=160$
>- $B$ computes $K=(Y_{A})^{X_{B}} \text{ mod } 353=40^{233}\text{ mod }353=160$
>
>>[!danger] Attack
>>The attacker must solve $3^\alpha \text{ mod }353=40$ or $3^b\text{ mod }353=248$ which are hard

>[!example] Man-in-the-middle attack
>- Darth generates a private key $X_{D_{1}}$ and $X_{D_{2}}$, and their public keys $Y_{D_{1}}$ and $Y_{D_{2}}$
>- Alice transmits $Y_{A}$ to Bob
>- Darth intercepts $Y_{A}$ and transmits $Y_{D_{1}}$ to Bob. Darth also calculates $K_{2}$
>- Bob receives $Y_{D_{1}}$ and calculates $K_{1}$
>- Bob transmits $X_{A}$ to Alice
>- Darth intercepts $X_{A}$ and transmits $Y_{D_{2}}$ to Alice. Darth calculates $K_{1}$
>- Alice receives $Y_{D_{2}}$ and calculates $K_{2}$
>- all subsequent communications are compromised

---
## Other public-key algorithms
- Digital Signature Standard (DSS)
	- makes use of SHA-1 and the Digital Signature Algorithm (DSA)
	- originally proposed in 1991, revised in 1993 due to security concerns, and another minor revision in 1996
	- cannot be used for encryption or key exchange
	- uses an algorithm that is designed to provide only the digital signature function
- Elliptic-Curve Cryptography (ECC)
	- equal security for smaller bit size than RSA
	- confidence level in ECC is not yet as high as that in RSA
	- based on a mathematical construct known as the elliptic curve
- El Gamal
	- based on the Diffie-Hellman secret- public key scheme
	- its security depends on the difficulty of computing discrete logarithms

---
## Post-quantum cryptography
Growing concern: future developments in quantum computers may enable the efficient solution of the hard problems of public key
schemes. For this reason NIST started a project to identify and standardize algorithms that resist future cyberattacks

In NISTIR 8413, they announced the selection of the first four such algorithms:
- CRYSTALS–KYBER
- CRYSTALS–Dilithium
- FALCON
- SPHINCS