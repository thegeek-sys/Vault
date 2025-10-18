---
Class: "[[Cybersecurity]]"
Related:
  - "[[Authentication]]"
---
---
## Introduction
In the password based authentication, the user provides name/login and password, so that the system can compare the password with the one stored for that specified login

The user ID:
- determines that the user is authorized to access the system
- determines the user’s privileges

>[!info] Password vulnerabilities
>![[Pasted image 20251014151426.png]]

>[!tip] Social engineering
>The social engineering attempts to use various psychological conditions in humans to get hold of confidential information
>
>It can be made in three possible ways:
>- *pretexting* → creating a story that convinces an administrator or operator into revealing secret information
>- *baiting* → offering a king of “gift” to get a user or agent to perform an insecure action
>- *quid pro quo* → offering an action or service then expecting something in return

---
## How is password stored?
The passwords are stored through a **cryptographic hash function** that outputs a checksum on messages of any length. The output is of a constant, fixed size, independent from the input length.

An hash function to be considered safe has to be:
- impossible to invert
- very efficient to compute
- very hard to fin two input values with the same output

>[!question] Why don’t we use encryption?
>Encryption is not used because we don’t want anyone to be able to revert it.

### UNIX-style, legacy
It uses up to 8 printable character in length while a 12 bit *salt* is used to modify DES encryption into a one-way hash function (to make sure that two different have two different hash even if the password is the same)

Once you have `password + salt` it is encrypted 25 times using DES with seed $0$, to get an output that consists of $11$ characters

Now is regarded as inadequate, even if it is still often required for compatibility with existing account management software or multivendor environments

### UNIX-style, today
#### UNIX
Unix uses a salt of up to 48 bits encrypted 1000 times with **MD5** crypt routine. It has no limitation on password length and produces a 128 bit hash value
#### OpenBSD
It uses *Bcrypt*, a hash function based on the Blowfish symmetric block cipher that allows passwords of up to 55 characters in length and requires a random salt value of 128 bits
It produces a 192 bit hash value and also includes a configurable cost variable to increase the time required to perform a Bcrypt hash (administrators can assign higher cost to privileged users)

---
## Strong passwords
