---
Created: 2025-09-29
Class: "[[Cybersecurity]]"
Related:
---
---
## Definition
The NIST definition says:

>[!quote]
>Prevention of damage to, protection of, and restoration of computers, electronic communications systems, electronic communications services, wire communication, and electronic communication, including information contained therein, to ensure its availability, integrity, authentication, confidentiality, and nonrepudiation.

### Computer security
While **Computer Security** are the measures and controls that ensure *confidentiality*, *integrity* and *availability* of information system **assets**, including hardware, software, firmware and information being processed, stored and communicated

---
## C.I.A.
The key concepts of security stands in the acronym C.I.A.
### Confidentiality
Preserving authorized restrictions on information access and disclosure, including means for protecting personal privacy and proprietary information

>[!example] Data + privacy

We can resume it as the avoidance of the unauthorized disclosure of information (involves protection of date, providing access for those who are allowed to see it while disallowing others from learning anythimg about its content)

#### Tools
##### Encryption
	The transformation of information using a secret, called an ecryption key, so that the transformed information can only be read using another secret, called the decryption key (which may, in some cases, be the same as the encryption key)

### Integrity
Guarding against improper information modification or destruction, including ensuring information nonrepudiation and authenticity

>[!example] Data integrity + system integrity

### Availablity
Ensuring timely and reliable access to and use of information

### Levels of impact
Depending on how much the system is compromised we have three levels of impact
- Low → the loss could be expected to have a limited adverse effect on organizational operations, organizational assets, or individuals
- Moderate → the loss could be expected to have serious adverse effect on organizational operations, organizational assets, or individuals
- High → the loss could be expected to have a sever or catastrophic adverse effect on organizational operations, organizational assets, or individuals

---
## Computer security challenges
1. Computer security is not as simple as it might first appear to the novice
2. In developing a particular security mechanism or algorithm, one must always considera potential attacks on those security features
3. Procedures used to provide particular services are ofter counterintuitive
4. Physical and logical placement needs to be detected
5. Security mechanisms typically involve more than a particular algorithm or protocol and also require that participants be in possession of some secret information which raises questions about the creation, distribution and protection of that secret information
6. Attackers only need to find a single weakness, while the designer must find and eliminate all weaknesses to achieve protect security
7. Security is still too ofter an afterthought to be incorporated into a system after the design is complete, rather than being an integral part of the design process
8. Security requires regular and constant monitoring
9. There is a natural tendency on the part of the users and system managers to perceive little benefit from security until a security failure occurs
10. Many users and even security administrators view string security as in impediment to efficient and user-friendly operation of an information system or use of information

---
## Assets
The **asset** is a key concept. In fact te assets are the things that are important for a person, a company or an institution, to be protected

>[!example] Examples
>- Staff address book
>- Patient records
>- Equipments
>- Criminal records
>- Keys for net-banking

---
## Vulnerabilities, threats and attacks
![[Pasted image 20250929151258.png|center|400]]

- **Categories of vulnerabilities**
	- corrupted → loss of integrity
	- leaky → loss of confidentiality
	- unavailable or very slow → loss of availability
- **Threats**
	- capable of exploiting vulnerabilities
	- represent potential security harm to an asset
- **Attacks** (threats carried out)
	- *passive* → attempt to learn of make use of information from the system that does not affect system resources (obtaining information that is being transmitted)
	- *active* → attempt to alter system resources or affect their operation (involves some modification of the data stream or the creation of a false stream)
	- *insider* → initiated by an entity inside the security perimeter. The insider is authorized to access system resources but uses them in a way not approved by those who granted the authorization
	- *outsider* → initiated from outside the perimeter by an unauthorized or illegitimate user of the system

>[!info] Security concepts
>- Threat agent → who conducts or has the intent to conduct detrimental activities
>- Countermeasure → a device or techniques that has as its objective  the impairment adversarial activity
>- Risk → a measure of the extent to which an entity is threatened by a potential circumstance or event
>- Threat → any circumstance or event with the potential to adversely impact organizational operations
>- Vulnerability → weakness in an information system, system security procedures, internal controls, or implementation that could be exploited or triggered by a threat source

