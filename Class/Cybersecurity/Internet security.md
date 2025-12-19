---
Class: "[[Cybersecurity]]"
Related:
---
---
## MIME and S/MIME
### MIME
MIME is an extension to the old RFC 822 specification of an Internet mail format (RFC 822 defines a simple heading  with `To`, `From`, `Subject`)

It provides a number of new header fields that define information about the body of the message

### S/MIME
S/MIME (Secure/Miltipurpose Internet Mail Extension) is a security enhancement to the MIME Internet e-mail format (based on technology from RSA Data Security)

It provides the ability to sign and/or encrypt e-mail messages

![[Pasted image 20251219113721.png]]

#### Functions
- enveloped data → encrypted content and associated keys
- signed data → encoded message + signed digest
- clear-signed data → cleartext message + encoded signed digest
- signed and enveloped data → nesting of signed and encrypted entities

#### Sign + encrypt
![[Pasted image 20251219114110.png|500]]

#### Decrypt + verify
![[Pasted image 20251219114203.png|500]]

#### Signed and Clear-Signed data
The preferred algorithms used for signing S/MIME messages use either an RSA os a DSA signature of a SHA-256 message hash

The process works as follows:
- take the message you want to send and map it into fixed-length code of 256 bits using SHA-256
- the 256 bit message digest is unique for this message making it virtually impossible for someone to alter this message or substitute another message and still come up with the same digest
- S/MIME encrypts the digest using RSA and the sender’s private RSA key
- the result is the digital signature, which is attacked to the message

Now anyone who hets the message can recompute the message digest then decrypt the signature using RSA and the sender’s public RSA key.
Since this operation only involves encrypting and decrypting a 256-bit block, it take up little time

#### Evenloped data
Default algorithms used for encrypting S/MIME message are AES and RSA

