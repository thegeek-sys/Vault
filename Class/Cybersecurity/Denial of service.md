---
Class: "[[Cybersecurity]]"
Related:
---
---
## Introduction

>[!quote] NIST Computer Security Incident Handling Guide definition of a DoS attack
>”An action that prevents or impairs the authorized use of networks, systems, or applications by exhausting resources such as central processing units (CPU), memory, bandwidth, and disk space.”

Denial-of-Service (DoS) is a form of attack on the availability of some service. The categories of resources that could be attacked are:
- *network* (bandwidth) → relates to the capacity of the network links connecting a server to the network; for most organizations this is their connection to their ISP
- *system* (resources) → aims to overload or crash the network handling software
- *application* (resources) → typically involves a number of valid requests, each of which consumes significant resources, thus limiting the ability of the server to respond to requests from other users

>[!example]
>![[Pasted image 20251125213527.png|450]]

---
## Classic DoS attacks
### Flooding ping
The aim of this attack is to overwhelm the capacity of the network connection to the target organization, in particular, even tho this traffic can be handled by higher capacity links on the path, as they reach links with lower capacity the packets are discarded and the network performance is noticeably affected.

Source of the attack is clearly identified unless a spoofed address is used

### Source address spoofing
Attacker use a forged source address, usually via the raw socket interface on operating systems to make the attack harder to identify.
Then the attacker generates large volumes of packets that have the target system as the destination address (but different origin address).

Congestion would result in the router connected to the final, lower capacity link, and this attack will require network engineers to specifically query flow information from their routers

### SYN spoofing
This one of the most common DoS attack. It attacks the ability of a server to respond to future connection requests by overflowing the tables used to manage them (legitimate users are denied access to the server).

Hence an attack on system resources, specifically the network handling code in operating system

![[Pasted image 20251125215750.png|420]]

---
## Flooding attacks
Flooding attacks are classified based on the network protocol used. The intent is to overload the network capacity on some link to a server. Virtually any type of network packet can be used.

There are three kinds of this attack:
- ICMP flood → ping flood using ICMP echo request packets as traditionally network administrators allow such packets into their networks because ping is a useful network diagnostic tool
- UDP flood → uses UDP packets directed to some port number on the target system
- TCP SYN flood → sends TCP packets to the target system; total volume of packets is the aim of the attacker rather than the system code

---
## Distributed Denial of Service (DDoS) attacks
DDoS attacks consist in the use of multiple systems to generate attacks. The attacker uses a flaw in operating system or in common application to gain access and install their program on it (zombie)

Large collection of such systems under the control of one attacker can be created, forming a *botnet*

![[Pasted image 20251125221608.png|470]]

### Mirai short story
In September 2016, the authors of the Mirai malware launched a DDoS attack on the website of a well-known security expert ([krebsonsecurity.com](https://krebsonsecurity.com)). A week later they released the source code into the world, possibly in an attempt to hide the origins of the attack

The code was quickly replicated by other cybercriminals, and is believed to be behind the massive attack that brought down the domain registration services provider, Dyn, in October 2016

### Examples of DDoS attacks
The biggest DDoS attack to date happened in September 2017, when the Google services received an attack with size of 2.54 Tbps. It was a reflection attack: spoofed packets sent to 180.000 web servers, which in turn sent responses to Google

In February 2020, AWS saw incoming traffic at a rate of 2.3 Tbps. The attacker responsible used hijacked Connection-less Lightweight Directory Access Protocol (CLDAP) web servers

In February 2018, an attack against Github reached 1.3 Tbps, sending packets at a rate of 126.9 million per second

In May 2025, Cloudflare mitigated the largest distributed denial-of-service (DDoS) attack ever reported, at 7.3 Tbps. The attack lasted around 45 seconds, and in that time delivered about 37.4 Tb of attack traffic
The 7.3 Tbps attack used multiple attack vectors against its target, including UDP floods, NTP reflection attacks, and traffic from the Mirai botnet. Fortunately, the Cloudflare network was able to automatically mitigate the attack and keep the target, a hosting provider, from falling victim.

---
## SIP invite scenario
![[Pasted image 20251125222542.png|400]]

---
## Hypertext Transfer Protocol (HTTP) based attacks
### HTTP flood
HTTP flood is an attack that bombards Web servers with HTTP requests (LOIC, HOIC) and consumes considerable resources

We talk about *spidering* when bots starts from a given HTTP link and by follows all links on the provided website in a recursive way

### Slowloris - R.U.D.Y.
Slowloris attack attempts to monopolize a website by sending HTTP requests that nevel complete and eventually consumes web server’s connection capacity.

It uses legitimate HTTP traffic as existing intrusion detection and prevention solution rely on signatures to detect attacks, so they generally won’t recognize Slowloris

---
## Reflection attacks
In this case the attacker sends packets to a known service on the intermediary with a spoofed source address of the actual target system and when the intermediary responds, the response is sent to the target, “reflecting” the attack off the intermediary (reflector)

The goal is to generate enough volumes of packets to flood the link to the target system without alerting the intermediary. The basic defense against these attacks is blocking spoofed-source packets

>[!example] DNS reflection attack
>![[Pasted image 20251125223515.png|500]]

### DNS amplification attacks
Amplification attacks use packets directed at a legitimate DNS server as the intermediary system. The attacker creates a series of DNS requests containing the spoofed source address of the target system and exploits the DNS behavior to convert a small request to a much larger response (amplification), so the target is flooded with responses.

![[Pasted image 20251125223822.png|500]]

Basic defense against this attack is to prevent the use of spoofed source address

>[!example] Amplification example
>Request: 64 bytes
>```
>dig ANY isc.org @x.x.x.x
>```
>
>Which results in a 50x amplification

---
## Memcached DDoS attack
Memcached is a high-performance caching mechanism for dynamic websites, that allows to speed up the delivery of web contents. The idea is to make a request 