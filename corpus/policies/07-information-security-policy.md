---
doc_id: INFOSEC
title: Information Security Policy
version: 5.0
effective_date: 2026-01-05
owner: IT Security
applies_to: All employees and contractors
---

# Information Security Policy

## SEC-1 Purpose and Scope

This policy establishes baseline security requirements for all individuals accessing Northbridge systems, data, or client information, regardless of employment type or work location. Client contracts may impose additional security requirements, which are communicated to relevant project teams separately and take precedence over this policy where stricter; engagement leads are responsible for making sure their project team is aware of any client-specific requirements before the engagement begins.

This policy applies uniformly across onsite, hybrid, and remote work arrangements as defined in RW-1; a device or account is subject to the same requirements whether it is used from a Company office or from an approved remote location.

## SEC-2 Acceptable Use

Company systems and accounts are provided for business use. Incidental personal use is permitted as long as it does not interfere with work, consume significant resources, or involve illegal, discriminatory, or harassing content. Employees must not share their account credentials with anyone, including coworkers or IT staff, and must not use another employee's credentials. IT staff will never ask an employee for their password; any such request, even one that appears to come from an internal address, should be treated as a phishing attempt and reported under SEC-6.

## SEC-3 Data Classification

Northbridge classifies data into three tiers: **Public** (no restriction on disclosure, such as published marketing material), **Internal** (Company business information not intended for external release, such as internal financial reports or org charts), and **Confidential** (client data, employee personal information, and trade secrets). Confidential data must not be stored on personal devices or personal cloud storage, must be encrypted at rest and in transit, and access is limited to individuals with a documented business need. Employee PTO balances, benefits elections, and HR ticket contents are all treated as Confidential data under this classification, consistent with the confidentiality expectations described in HR-OPS-4.

When an employee is uncertain which tier a particular piece of information falls into, the safer assumption is Confidential until confirmed otherwise, particularly for anything involving a named individual (employee or client contact) or anything not already published externally.

## SEC-4 Remote and Travel Security Requirements

Employees working remotely, including under an approved temporary or international remote work arrangement (see RW-3 and RW-4), must connect to Company systems only over the Company VPN when accessing Confidential data, must use full-disk encryption on all Company devices (enabled by default on Company-issued equipment per EQ-1), and must not access Confidential data over unsecured public Wi-Fi without the VPN active. When traveling internationally, employees should assume that some countries may inspect device contents at the border; IT Security can provide a loaner device for high-risk travel on request, coordinated through an HR ticket (see HR-OPS-2). A loaner device carries minimal locally stored data and relies on VPN-based access to Company systems, reducing the amount of Confidential data physically present on the device during border crossings.

## SEC-5 Access Control and Authentication

All Company accounts require multi-factor authentication. Access to client systems and Confidential data follows least-privilege principles and is reviewed quarterly by IT Security and relevant engagement leads. Access for departing employees is revoked on or before their last day of employment, coordinated between IT and People Operations as part of offboarding, so that no gap exists between an employee's last working day and the removal of their system access.

## SEC-6 Incident Reporting

Any suspected security incident — including a lost or stolen device, a phishing attempt that was acted on, or suspected unauthorized access — must be reported to IT Security within 24 hours through the security incident channel or an urgent HR ticket. Prompt reporting is not subject to disciplinary action even where employee error contributed to the incident; delayed or concealed reporting may be addressed under the Workplace Conduct policy (CONDUCT-1). The goal of this section is to make early reporting the clearly safer choice for an employee in any incident scenario, since faster reporting materially reduces the potential impact of most security incidents.

## SEC-7 Third-Party and AI Tool Use

Use of third-party AI tools (including code assistants, writing assistants, and chat-based tools not provided or approved by Northbridge IT) with Confidential data, including client data or employee personal information, is prohibited unless the specific tool has been reviewed and approved by IT Security and added to the approved tools list. Company-approved AI tools may be used with Internal and Public data following standard acceptable-use practices in SEC-2. This restriction exists because many third-party AI tools process submitted data on external servers in ways that may not meet the encryption, access control, or data residency expectations described elsewhere in this policy, and because client contracts frequently include specific restrictions on where and how client data may be processed.

Employees uncertain whether a specific AI tool is approved should check the current approved tools list in the internal IT portal before using it with any Internal or Confidential data, rather than assuming a popular or widely used tool is automatically approved.

## SEC-8 Consequences of Violations

Violations of this policy may result in suspension of system access pending investigation, and may lead to disciplinary action up to and including termination, consistent with the Workplace Conduct policy (CONDUCT-1). Client contract violations arising from a security incident may also result in financial or reputational consequences to the Company that are considered in determining the appropriate response. As with SEC-6, a violation that is promptly self-reported is generally treated more favorably than one discovered independently, though self-reporting does not eliminate the possibility of consequences for a serious or repeated violation.

## SEC-9 Password and Authentication Standards

Company accounts require a minimum password length of 14 characters or use of a Company-approved password manager with a strong generated password, in addition to the multi-factor authentication requirement in SEC-5. Passwords must not be reused across Company and personal accounts. IT Security may require a password reset following a suspected or confirmed incident under SEC-6, independent of any scheduled rotation, and employees should complete a required reset promptly since delaying it leaves the potentially affected account at elevated risk in the interim.

## SEC-10 Relationship to Remote Work and Tax Location Policies

Because approved remote and international work arrangements under RW-3 and RW-4 already involve an IT Security review as part of the approval process in RW-6, an employee who receives approval for a location change does not need to separately re-confirm compliance with this policy for that specific trip beyond following the VPN, encryption, and reporting requirements described above; the location-change approval process is designed to surface any location-specific security concerns (such as heightened data residency requirements in a particular country) before the trip begins, rather than leaving employees to independently assess those risks.

## SEC-11 Frequently Asked Questions

**Can I check work email from my personal phone without enrolling it in mobile device management?** No. Email and calendar access from a personal device requires enrollment per EQ-4, which enforces a passcode and remote-wipe capability; this applies even for brief or occasional access.

**Is it acceptable to save a client spreadsheet to my personal Google Drive so I can access it from home?** No. Client data is Confidential under SEC-3 and must not be stored on personal cloud storage, regardless of convenience; use the Company VPN and Company-managed storage to access the file from home instead.

**I used a free online AI writing tool to help draft an internal email that didn't contain client data. Was that allowed?** Internal, non-Confidential drafting content can generally be used with an approved AI tool under SEC-7; if the specific tool is not on the approved list, check with IT Security before repeating this, since the approved-tools list exists precisely to make this determination in advance rather than case by case.

**I think I clicked a phishing link but didn't enter any credentials. Should I still report it?** Yes. Report it within 24 hours under SEC-6 regardless of whether you believe any harm occurred; IT Security can only assess and contain a potential issue if they know about it promptly.
