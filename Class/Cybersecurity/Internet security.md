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
