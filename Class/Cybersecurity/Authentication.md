---
Class: "[[Cybersecurity]]"
Related:
---
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

---
## Authentication
The four means of authenticating user identity are based on:
- something the individual *knows*
- something the individual *possesses*
- something the individual *is*
- something the individual *does*

### Multifactor authentication
Multifactor authentication is the use of more than one of the authentication means and using more factors is considered stronger than using less

---
## Assurance Levels for user authentication
An organization can choose from a range of authentication technologies, based on the degree of confidence in identity proofing and authentication processes
### IAL
There are three levels of **Identity Assurance Levels** (*IAL*):
- **IAL 1** → no need to link the applicant to a specific real-time identity
- **IAL 2** → provides evidence for the claimed identity using either remote or physically-present identity proofing