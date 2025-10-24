---
Class: "[[Cybersecurity]]"
Related:
---
---
## Index
- [[#Introduction|Introduction]]
	- [[#Introduction#Authentication|Authentication]]
		- [[#Authentication#Multifactor authentication|Multifactor authentication]]
	- [[#Introduction#Assurance Levels for user authentication|Assurance Levels for user authentication]]
		- [[#Assurance Levels for user authentication#IAL|IAL]]
		- [[#Assurance Levels for user authentication#AAL|AAL]]
- [[#Passwords|Passwords]]
- [[#How is password stored?|How is password stored?]]
	- [[#How is password stored?#UNIX-style, legacy|UNIX-style, legacy]]
	- [[#How is password stored?#UNIX-style, today|UNIX-style, today]]
		- [[#UNIX-style, today#UNIX|UNIX]]
		- [[#UNIX-style, today#OpenBSD|OpenBSD]]
- [[#Strong passwords|Strong passwords]]
	- [[#Strong passwords#Password cracking|Password cracking]]
		- [[#Password cracking#Dictionary attacks|Dictionary attacks]]
		- [[#Password cracking#Rainbow table attacks|Rainbow table attacks]]
	- [[#Strong passwords#Password selection strategies|Password selection strategies]]
- [[#Tokens|Tokens]]
	- [[#Tokens#Authentication via barcodes|Authentication via barcodes]]
	- [[#Tokens#Magnetic stripe cards|Magnetic stripe cards]]
	- [[#Tokens#Smart tokens|Smart tokens]]
		- [[#Smart tokens#Smart cards|Smart cards]]
			- [[#Smart cards#eIDs|eIDs]]
	- [[#Tokens#One-time password (OTP) device|One-time password (OTP) device]]
		- [[#One-time password (OTP) device#Time-bases one-time password (TOTP)|Time-bases one-time password (TOTP)]]
	- [[#Tokens#Hardware authentication token pros-n-cons|Hardware authentication token pros-n-cons]]
- [[#Biometrics|Biometrics]]
	- [[#Biometrics#Accuracy dilemma|Accuracy dilemma]]
	- [[#Biometrics#Security vs. convenience|Security vs. convenience]]
	- [[#Biometrics#Operation of a biometric authentication system|Operation of a biometric authentication system]]
- [[#Remote user authentication|Remote user authentication]]
	- [[#Remote user authentication#Basic challenge-response protocols: password and token|Basic challenge-response protocols: password and token]]
- [[#Authentication security issues|Authentication security issues]]
---
## Introduction

>[!quote] NISP SP 800-64-4 (Digital Authentication Guideline, October 2024)
>”The process of establishing confidence in user identities that are presented electonically to an information system”

This definition give some identification and authentication requirements for protecting data:
- uniquely identify and authenticate system users, and associate that **unique identification** with processes acting of behalf of those users (reauthenticate users when needed)
- implement **multi-factor authentication** for access to privileged and non-privileged accounts
- implement **replay-resistant authentication** mechanisms for access to privileged and non-privileged accounts
- **identifier management**
	- receive authorization from organizational personnel or roles to assign an individual, group, role, service, or device identifier
	- select and assign an identifier that identifies an individual, group, role or device
	- prevent the reuse of identifiers for a defined time period
	- manage individual identifiers by uniquely identifying each individual characteristic
- **password management**
	- maintain a list of commonly-used, expected, or compromised passwords, and frequently update the list, even when organizational passwords are suspected to have been compromised
	- verify that passwords are not found on the list of commonly used, expected, or compromised passwords when users create or update passwords
	- transmit passwords only over cryptographically protected channels
	- store passwords in a cryptographically protected form
	- select a new password upon first use after account recovery
	- enforce composition and complexity rules for passwords.
- obscure **feedback of authentication** information during the authentication process
- **authenticator management** (the element that allows to perform the authentication passwords, biometrics, …)
	- verify the identity of the individual, group, role, service, or device receiving the authenticator as part of the initial authenticator distribution
	- establish initial authenticator content for any authenticators issued by the organization
	- establish and implement administrative procedures for initial authenticator distribution; for lost, compromised, or damaged authenticators; and for revoking authenticators
	- change default authenticators at first use
	- change or refresh authenticators frequently or when relevant events occur
	- protect authenticator content from unauthorized disclosure and modification

>[!info] NIST SP 800-63-3 Digital Identity Guidelines architecture model
>![[Pasted image 20251014145201.png]]
### Authentication
The four means of authenticating user identity are based on:
- something the individual *knows*
- something the individual *possesses*
- something the individual *is*
- something the individual *does*

#### Multifactor authentication
Multifactor authentication is the use of more than one of the authentication means and using more factors is considered stronger than using less

### Assurance Levels for user authentication
#### IAL
An organization can choose from a range of authentication technologies, based on the degree of confidence in identity proofing and authentication processes

There are three levels of **Identity Assurance Levels** (*IAL*):
- *IAL 1* → no need to link the applicant to a specific real-time identity
- *IAL 2* → provides evidence for the claimed identity using either remote or physically-present identity proofing
- *IAL 3* → requires physical presence for identity proofing

#### AAL
AALs define options an organization can select, based on their risk assessment and the potential harm caused by an attacker taking control of an authenticator and accessing their system

There are three levels of **Authenticator Assurance Levels** (*AAL*):
- *AAL 1* → provides some assurance of authentication via user-supplied ID and password
- *AAL 2* → provides high confidence of authentication via proof or possession and control of two authentication factors
- *AAL 3* → provides very high confidence of authentication via proof of possession and control of two authentication factors

---
## Passwords
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

>[!example] A fixed 6 symbols password
>- numbers → $10^6=1.000.000$
>- upper or lower case characters → $26^6=308.915.776$
>- upper and lower case characters → $52^6=19.770.609.664$
>- 94 practical symbols available → $94^6=689.869.781.056$

The main problem is that passwords have to be stored in human memory and they tend to be easy (far from being random). For this reason password crackers use dictionaries of words

>[!example] Password does not change for 60 days, so how many passwords should I try for each second?
>- 5 characters → 1,415 PW/sec
>- 6 characters → 133,076 PW/sec
>- 7 characters → 12,509,214 PW/sec
>- 8 characters → 1,175,866,008 PW/sec
>- 9 characters → 110,531,404,750 PW /sec

### Password cracking

>[!info] NIST 800-63-4 Digital Identity Guidelines
>Proposed guidelines aim to inject badly needed common sense into password hygiene
>
>>[!example]
>>- verifiers and CSPs shall not impose other composition rules for passwords
>>- verifiers and CSPs shall not require users to change passwords periodically (but they shall force a change if there is evidence of compromise of the authenticator)
>
>When passwords are chosen properly, the requirement to periodically change them, typically every one or three months, can actually diminish security because the added burden incentivizes weaker passwords that are easier for people to set and remember
>
>Other password criteria:
>- verifiers and CSPs SHALL require passwords to be a minimum of eight characters in length and SHOULD require passwords to be a minimum of 15 characters in length.
>- verifiers and CSPs SHOULD permit a maximum password length of at least 64 characters.
>- verifiers and CSPs SHOULD accept all printing ASCII characters and the space character in passwords.
>- verifiers and CSPs SHOULD accept Unicode characters in passwords (each Unicode code point shall be counted as a single character when evaluating password length).
>- verifiers and CSPs SHALL NOT permit the subscriber to store a hint that is accessible to an unauthenticated claimant.
>- verifiers and CSPs SHALL NOT prompt subscribers to use knowledge-based authentication (KBA) (e.g. “What was the name of your first pet?”) or security questions when choosing passwords.
>- verifiers SHALL verify the entire submitted password (not truncate it).

Typically password crackers exploit the fact that people choose easily guessable passwords (shorter password lengths are also easier to crack). 
One of the most famous is **John the Ripper**, an open-source password cracker first developed in 1996; it uses a combination of brute-force and dictionary techniques.

There are two possible ways of exploiting a password:
- dictionary attacks
- rainbow table attacks

#### Dictionary attacks
Consists of developing a large dictionary of possible passwords and try each against the password file. Each password must be hashed using each salt value and then compared to the stored hash values

#### Rainbow table attacks
It consists of precompiting tables of hash values for all salts (a *mammoth table* of hash values)
It can be countered by using sufficiently large salt value and a sufficiently large hash length

### Password selection strategies
Many systems nowadays have a complex password policy (*proactive password checking*). With this policy users are allowed to select their own password, however the system checks to see if the password is allowable, and if not, rejects it (if the password is weak it is not accepted).

The goal is to eliminate guessable passwords while allowing the user to select a password that is memorable. A disadvantage is the space required by dictionaries and the time to check

---
## Tokens
Other than passwords there are many possible ways to authenticate like through **authentication tokens**:
- memory cards
- barcodes
- magnetic stripe cards
- smart cards (contact, contactless)
- RFIDs

### Authentication via barcodes
Boarding passes, which are created at flight check-in and scanned before boarding use barcodes

The barcode in this case encodes an internal unique identifier that allows airport security to look up the corresponding passenger’s record

In most other applications, however, barcodes provide convenience but not security (since barcodes are simply images, they are extremely easy to duplicate)

### Magnetic stripe cards
Those are plastic cards with a magnetic stripe containing personalized information about the card holder

![[Pasted image 20251020150845.png|300]]

The first track of a magnetic stripe card contains the cardholder’s full name in addition to an account number, format information, and other data. The second track may contain the account number, expiration date, information about the issuing bank, data specifying the exact format of the track, and other discretionary data.

One vulnerability of the magnetic stripe medium is that it is easy to read and reproduce, in fact magnetic stripe readers can be purchased at relatively low cost, allowing attackers to read information off cards. When couples with magnetic stripe writer, an attacker can easily clone existing cards.
For this reason, many uses require card holders to enter a PIN to use their cards.

### Smart tokens
Smart tokens unlike the other tokens, include an embedded microprocessor and typically have manual interfaces for human/token interaction.

A smart card or other token requires an electronic interface to communicate with a compatible reader/writer (contact and contactless interfaces)

Their authentication protocol is classified into three categories:
- static
- dynamic password generator
- challenge-response

#### Smart cards
These are the most important category of smart token. It contain an entire microprocessor with processor, memory, I/O ports

Typically include three types of memory:
- read-only memory (ROM) → stores data that does not change during the card’s life
- electrically erasable programmable ROM (EEPROM) → holds application data and programs
- random access memory (RAM) → holds temporary data generated when applications are executed

##### eIDs
A national electronic identity (eID) card can serve the same purposes as other national ID cards. In addition, an eID card can provide stronger proof of identity and be uses in a wider variety of applications:
- *ePass* → a digital representation of the cardholder’s identity (ig. electronic passport)
- *eID* → an identity record that authorized service can access with cardholder permission
- *eSign* → this optional function stores a private key and a certificate verifying the key; it is used for generating a digital signature

Human-readable data are printed on its surface

>[!info] Password Authenticated Connection Establishment
>The **Password Authenticated Connection Establishment** (*PACE*) ensures that the contactless RF chip in the eID cannot be read without explicit access control.
>
>For online applications, access is established by the user entering the six-digit PIN (which should be known only to the holder of the card)
>For offline applications, either the MRZ printed on the back of the card or the six-digit access number (CAN) printed on the front is used
>
>![[Pasted image 20251020162917.png]]

### One-time password (OTP) device
It has a secret key to generate an OTP which is then validated by the system. 

It uses a block cipher/hash function to combine secret key and time or nonce value to create OTP and has a tamper-resistant module for secure storage of the secret key

#### Time-bases one-time password (TOTP)
TOTP uses HMAC (mode of message authentication) with a hash function. It is used in many hardware token and by many mobile authenticator apps.

The password is computed from the current Unix format time value and system using time based OTP need to allow for clock drift between token and verifying system while systems using nonce need to allow for failed authentication attempts

>[!info] Using SMS as OTP
>Pros:
>- one of the simplest authentication approaches
>- no need to have any additional app on the phone
>
>Cons:
>- requires mobile coverage to receive SMS
>- when mobile phone is lost or stolen, user will lose access or an attacker might gain access
>- attackers might use a SIM swap attack or change the authenticated phone number
>- attacker might also intercept messages using either a fake mobile tower, or by attacking SS7 signaling protocol

### Hardware authentication token pros-n-cons
The main disadvantage is that any other person can see the code but for this reason it is only used in multifactor authentication

## Biometrics
Biometric refers to any measure used to uniquely identify a person based on biological or physical traits. A biometric system incorporate a sensor or scanner to read biometric information and then compare this information to stored templates of accepted users before granting access

It is based on pattern recognition but is technically complex and expensive when compared to passwords tokens

![[Pasted image 20251020164757.png|400]]

### Accuracy dilemma
In this depiction, the comparison between the presented freature and a reference feature is reduces to a single numeric value

![[Pasted image 20251024161156.png|500]]

If the input value ($s$) is greater than a preassigned threshold ($t$), a match is declared

>[!example] Example for a face recognition system
>An instance of genuine and imposter score distributions for a face recognition system
>
>![[Pasted image 20251024161413.png|400]]

### Security vs. convenience
Idealized biometric measurement operating characteristic curves (log-log scale)

![[Pasted image 20251024161626.png|450]]

### Operation of a biometric authentication system
There are three steps:
- enrollment
- verification
- identification

![[Pasted image 20251024161739.png|450]]
![[Pasted image 20251024161753.png|450]]
![[Pasted image 20251024161800.png|450]]

---
## Remote user authentication
Authentication over a network, the Internet, or a communication link is more complex, in fact there are additional security threats such as: eavesdropping, capturing a password, replaying an authentication sequence that has been observed

Usually, to counter threats, the remote user authentication rely in some form of a *challenge-response*

### Basic challenge-response protocols: password and token
![[Pasted image 20251024162315.png]]
![[Pasted image 20251024162332.png]]

---
## Authentication security issues
There are many security issues related to authentication:
- **client attacks** → adversary attempts to achieve user authentication without access to the remote host or the intervening communications path
- **host attacks** → directed at the user file at the host where passwords, token passcodes, or biometric templates are stored
- **eavesdropping** → adversary attempts to learn the password by some sort of attack that involves the physical proximity of user and adversary
- **replay** → adversary repeats a previously captured user response
- **trojan horse** → an application or physical device masquerades as an authentic application or device for the purpose of capturing a user password, passcode, or biometric
- **denial-of-service** → attempts to disable a user authentication service by flooding the service with numerous authentication attempts


| Attacks                       | Authenticators             | Examples                                       | Typical defenses                                                                                                           |
| ----------------------------- | -------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| client attack                 | password                   | guessing, exhaustive search                    | large entropy; limited attempts                                                                                            |
|                               | token                      | exhaustive search                              | large entropy; limited attempts; theft of object requires presence                                                         |
|                               | biometric                  | false match                                    | large entropy; limited attempts                                                                                            |
| host attacks                  | password                   | plaintext theft, dictionary/exhaustive search  | same as password; 1-time passcode                                                                                          |
|                               | token                      | passcode theft                                 | capture device authentication; challenge response                                                                          |
|                               | biometric                  | template theft                                 | capture device authentication; challenge response                                                                          |
| eavesdropping, theft, copying | password                   | “shoulder surfing”                             | user diligence to keep secret; administrator diligence to quickly revoke compromised passwords; multifactor authentication |
|                               | token                      | theft, counterfeiting hardware                 | multifactor authentication; tamper resistant/evident token                                                                 |
|                               | biometric                  | copying (spoofing) biometric                   | copy detection at capture device and capture device authentication                                                         |
| replay                        | password                   | replay stolen password response                | challenge-response protocol                                                                                                |
|                               | token                      | replay stolen passcode response                | challenge-response protocol; 1-time passcode                                                                               |
|                               | biometric                  | replay stolen biometric template response      | copy detection at capture device and capture device authentication via challenge-response protocol                         |
| trojan horse                  | password, token, biometric | installation of rogue client or capture device | authentication of client or capture device within  trusted security perimeter                                              |
| denial of service             | password, token, biometric | lockout by multiple failed authentication      | multifactor with token                                                                                                     |
