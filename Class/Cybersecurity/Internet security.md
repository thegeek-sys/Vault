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

It operates as follows:
- S/MIME generates a pseudorandom secret key that it used to encrypt the message using AES or some other conventional encryption scheme
- a new pseudorandom key is generated for each new message encryption
- this session key is bound to the message and transmitted with it
- the secret key is used as input to the public-key encryption algorithm, RSA, which encrypts the key with the recipient’s public RSA key
- on the receiving end, S/MIME uses the receiver’s private RSA key to recover the secret key, then uses the secret key and AES to recover the plaintext message
- if encryption is used alone, radix-64 is used to convert the ciphertext to ASCII format

---
## Domain Keys Identifier Mail (DKIM)
Domain Keys Identifier Mail is a specification of cryptographically signing e-mail messages permitting a signing domain (organization) to claim responsibility for a message in the mail stream (used to guarantee who is the sender)

It was firstly proposed in the Internet Standard (RFC 4871) and has been widely adopted by a range of e-mail providers

>[!info] Internet mail architecture (RFC 5598)
>![[Pasted image 20251219175604.png|500]]
>
>where:
>- MHS → message handling system
>- MTA → mail transfer agent
>- MUA → mail user agent
>- MDA → mail delivery agent
>- (E)SMTP → extended simple mail transfer protocol

DKIM is designed to provide an e-mail authentication technique that is transparent to the end user. A user’s e-mail message is signed by a private key of the administrative domain from which the e-mail originates.
The signature covers all of the content of the message and some message headers. At the receiving end, the MDA can access the corresponding public key via a DNS and verify the signature, thus authenticating that the message comes from the claimed administrative domain

![[Pasted image 20251219180458.png|400]]

A DKIM record stores the DKIM public key and e-mail servers query the domain’s DNS records to see the DKIM record and view the public key

>[!example] Example of DKIM record
>![[Pasted image 20251219180641.png|500]]

### S/MIME and DKIM comparison
S/MIME depends on both the sending and receiving users employing S/MIME. For almost all users, the bulk of incoming mail does not use S/MIME, and the bulk of the mail the user wants to send is to recipients not using S/MIME
S/MIME signs only the message content. Thus, header information concerning origin can be compromised

DKIM is not implemented in client programs (MUAs) and is therefore transparent to the user (the user need take no action). DKIM applies to all mail from cooperating domains and allows good senders to prove that they did send a particular message and to prevent forgers from masquerading as good senders

---
## Secure Sockets Layer (SSL) and Transport Layer Security (TLS)
SSL/TLS is one of the most widely used security service. It is a general-purpose service implemented as a set of protocols that rely on TCP, meaning that it can protect any kind of traffic that uses secure connection

TLS and SSL are the same protocol, but SSL is the original name of the protocol that was then standardized into TLS in Internet Standard (RFC 4346)

There are two implementation choices:
- provided as part of the underlying protocol suite (directly provided from the OS)
- embedded in applications

>[!info] SSL/TLS protocol stack
>![[Pasted image 20251219182150.png|400]]

>[!tldr] TLS concepts
>**TLS session** is an association between a client and a server, created by the handshake protocol. It defines a set of cryptographic security parameters and is used to avoid the expensive negotiation of new security parameters for each connection
>
>**TLS connection** is a transport (in the OSI layering model definition) that provides a suitable type of service providing transient peer-to-peer relationships where every connection is associated with one session

### TLS Record Protocol operations
![[Pasted image 20251219182731.png]]

The TLS record protocol provides two services:
- confidentiality → the Handshake Protocol defines a shared secret key that is used for symmetric encryption of TLS payloads
- message integrity → the Handshake Protocol also defines a shared secret key that is used to form a message authentication code (MAC)

The TLS record protocols is composed from 4 smaller protocols:
- **Change Cipher Spec Protocol**
- **Alert Protocol**
- **Handshake Protocol**
- **Heartbeat Protocol**

### Change Cipher Spec Protocol
It the the simplest protocol between the four. It consists of a single message composed of a single byte of 1s.

The sole purpose of this message is to cause pending state to be copied into the current state, to tell the receiver to stop using previous parameters and to start using a new cipher suite

### Alert Protocol
Alert Protocol conveys TLS-related alerts to peer entity. Alert messages are compressed and encrypted and each message consists of two bytes:
- first byte takes the value warning (1) or fatal (2) to convey the severity of the message; if the level is fatal, TLS immediately terminates the connection. Other connections on the same session may continue, but no new connections on this session may be established
- second byte contains a code that indicates the specific alert

### Handshake Protocol
It is the most complex part of TLS and is used before any application data are transmitted

It allows server and client to:
1. authenticate each other
2. negotiate encryption and MAC algorithms
3. negotiate cryptographic keys to be used

It comprises (consist of) a series of messages exchanged by client and server and has four phases

![[Pasted image 20251219184652.png|500]]

>[!hint]
>Shared transfers are optional or situation-dependent messages that are not always sent

### Heartbeat Protocol
It consists of a periodic signal generated by hardware or software to indicate normal operation or to synchronize other parts of a system.
It is typically used to monitor the availability of a protocol entity (to verify if the other is still online and working)

It runs on top of the TLS Record Protocol and the use is established during Phase 1 of the Handshake Protocol and each peer indicates whether it supports heartbeats. It serves two purpose:
- assures the send that the recipient is still alive
- generates activity across the connection during the idle period

### SSL/TLS attacks
There are four genera categories:
- attacks on the handshake protocol
- attacks on the record and application data protocols
- attacks on the PKI
- other attacks

>[!example] Heartbleed exploit
>![[Pasted image 20251219190310.png|400]]

### HTTPS
HTTPS (documented in RFC 2818) is the combination of HTTP and SSL to implement secure communication between a Web browser and a Web server and is built into all modern Web browsers

An agent acting as the HTTP client also as the TLS client. The closure of an HTTPS connection requires that TLS close the connection with the peer TLS entity on the remote side, which will involve closing the underlying TCP connection

---
## IP security
When we talk about IPSec, we mean a set of protocols born to protect communications directly at the network level. While other security mechanisms are specific for certain applications, IPSec aims to implements security directly in the network

IPSec was designed as an integral part of IPv6 natively including authentication and encryption functionalities, but it is still completely implemented in IPv4

Benefits:
- when implemented in a firewall or router, it provides strong security to all traffi crossing the perimeter
- in a firewall it is resistant to bypass
- below transport level, hence transparent to application
- can be transparent to end users
- can provide security for individual users
- secures routing architecture

IPSec offers basic function, provided by separate (sub-)protocols like:
- **Authentication Header** (AH) → support for data integrity and authentication of IP packers
- **Encapsulated Security Payloads** (ESP) → support for encryption and (optionally) authentication
- **Internet Key Exchange** (IKEv2) → support for key management etc.

>[!info]
>Because message authentication is provided by ESP, the use of AH is deprecated. It is included in IPSecv3 for backward compatibility but should not be used in new applications

### Transport and tunnel modes
Transport and tunnel mode are the two main modes in which IPSec operates. This modes define which part of the packet gets protected and how the packets travels the network
#### Transport mode
This mode is designed for a end-to-end protection between two host that communicate. It is used to protect the payload of the IP packet. In fact the IP header is not encrypted, to make the intermediate routers to read the destination address and to route the packet.

If you use the ESP protocol in this mode, it encrypts and optionally authenticates just the transferred data, leaving the IP header plain

#### Tunnel mode
Tunnel mode provides protection to the entire IP packet and it travels through a tunnel from one point to an IP network to another. It is used when one or both ends of a security association are a security gateway.

In this way the users in local networks behind a firewall can communicate in a safe way without implementing IPSec on the single hosts (the gateway creates the tunnel)

>[!info] Security associations
>If IPSec is the safe “tunnel”, the security association is the set of rules that defines how that tunnel has to be built and managed between the two hosts
>
>Security associations are one-way relationships between the sender and the receiver that affords security for traffic flow. If a peer relationship is needed for two-way secure exchange then two security associations are required
>
>A security association is uniquely identified by three parameters:
>- security parameter index (SPI)
>- IP destination address
>- protocol identifier
>
>![[Pasted image 20251220161349.png]]

---
## Virtual Private  Network

>[!quote] Definition (NIST SP800-113)
>”A virtual network, built on top of an existing network infrastructure, which can provide a secure communications mechanism for data and other information transferred between two endpoints”

A VPN is typically based on the use of encryption, but there are several possible choices for:
- how to perform encryption
- which parts of communication should be encrypted

>[!warning]
>If a solution is to difficult to use, it won’t be used. So the VPN should be easy to use to make the people use it

>[!info] Security goals
>- traditional
>	- confidentiality of data
>	- integrity of data
>	- peer authentication
>- extended
>	- replay protection
>	- access control
>	- traffic analysis protection

>[!info] Usability goals
>- transparency → VPN should be invisible to users, software, hardware
>- flexibility → VPN can be used between users, applications, hosts, sites
>- simplicity → VPN can be actually used

>[!abstract] Kinds of protection
>Site-to-site security
>![[Pasted image 20251221100030.png|450]]
>
>Host-to-site security
>![[Pasted image 20251221100046.png|450]]
>
>Host-to-host security
>![[Pasted image 20251221100124.png|450]]


>[!info] Firewall with an SSL VPN
>![[Pasted image 20251221102117.png|450]]
>
>The image shows how SSL VPN is the safe front door to link Internet with the private resources of a company, keeping a clear separation between public servicies (DMZ) and internal resources

### Tunneling
Tunneling is the operation of a network connection on top of another network connection. It allows two host or sites to communicate through another network that they do not want to use directly

#### Site-to-site tunneling
Site-to-site tunneling enables a PDU (Protocol Data Unit) to travel from a site to another without the content being read or processed by the intermediate nodes during the path

The secret for this is the **encapsulating**; the entire original packet is put inside another packet (PDU)

![[Pasted image 20251221104017.png|470]]

>[!info]
>The host-to-host communication does not need to use IP

#### Secure tunneling
![[Pasted image 20251221104731.png|470]]

The main difference between this and the previous method, is that this one, before encapsulating the original  packet, it encrypts it

### SSL VPN functionalities
Most SSL VPNs offer one or more core functionalities:
- proxying → intermediate device appears as true server to client (e.g. web proxy)
- application translation → conversion of information from one protocol to another (e.g. Portal VPN offers translation for applications which are not Web-enabled, so users can use Web browser to access applications with no Web interface)
- network extension: provision of partial or complete network access to remote users, typically via Tunnel VPN. There are two variants of this functionality:
	- full tunneling → all network traffic goes through tunnel
	- split tunneling → organization’s traffic goes through tunnel, other traffic uses remote user’s default gateway

### SSL VPN Security Services
Typical services include:
- authentication → via strong authentication methods, such as two-factor authentitcation, X.509 certificates, smartcards, security tokens etc. May be integrated in VPN device or external authentication server
- encryption and integrity protection → via the use of the SSL/TLS protocol
- access control → May be per-user, per-group or per-resource
- endpoint security controls → validate the security compliance of clients attempting to use the VPN (e.g. presence of antivirus system, updated patches etc.)
- intrusion prevention → evaluates decrypted data for malicious attacks, malware etc.

---
## Anonymity: Tor

>[!question] What is anonymity?
>Anonymity means that a person is not identifiable within a set of subjects and so are his actions (e.g. sender and his emails are no more related after adversary’s  observations than they were before)
>
>Unobservability → adversary cannot tell whether someone is using a particular system and/or protocol

>[!question] Why anonymity?
>- to protect privacy → avoid tracking by advertising companies, viewing sensitive content, information on medical conditions, advice on bankruptcy
>- protection from prosecution
>- to prevent chillin-effects → it’s easier to void unpopular or controversial opinions if you are anonymous

Anonymity on Internet is hard for many reasons:
- in every packet there is the source and destination IP address, and ISPs store communication records
- wireless traffic can be trivially intercepted
- tier 1 ASs (autonomous systems) and IXPs (internet exchange points) are compromised
- difficult if not impossible to achieve on your own 