# Prompts
## Human guided explanations
### System Instructions
You are an expert lawyer in the domain of U.S. civil procedure. You are given an introduction to a legal case, a question, an answer and an analysis that justifies the correctness of that answer. These are taken from a legal textbook, however they are not formal and they refer to multiple choice letters (A, B, C, D). Read the given analysis carefully and explain why the answer is correct in a short, formal, clear way, without mentioning anything about letters A, B, C, D. Reply with the format: "Explanation:\n<your_text>", in one sentence.
### Example
Introduction:<br>
Section 1391(d), which defines the residence of a corporation that has contacts in one district within a state but not others, is confusing. Suppose, for example, that Omni-Plex Corporation has its principal place of business in the Northern District of California, sufficient contact to support general in personam jurisdiction over the corporation, that is, jurisdiction for a claim that arises anywhere. Assume further that Omni- Plex has no contacts in any other federal district within California. Under §1391(d), Omni-Plex ‘‘resides’’ in the Northern District of California, because, if the Northern District were a state, its contacts there would be sufficient to support personal jurisdiction over it there. But it would not ‘‘reside’’ in the Eastern District of California. It has no contacts there, so that, if the Eastern District were a state, it would not be subject to personal jurisdiction there. This is confusing because a defendant that is ‘‘at home’’ in a state (Daimler A.G. at 137) is subject to personal jurisdiction anywhere in the state, not just in the part of the state where the contacts exist. But §1391(d) tells us that, for venue purposes, we should look at the contacts in each district within the state separately. The corporation will be deemed to ‘‘reside’’ only in the districts where its contacts would support personal jurisdiction if that district were a state. Here’s a question to illustrate the operation of this vexing provision.


Question:<br>
Manufacturing venue. Arthur wishes to bring a diversity action in federal court against Cleveland Manufacturing Company. Cleveland has its factory and principal place of business in the Northern District of Illinois, but no other contacts with Illinois. The claim is based on alleged negligence in making a toaster at the Illinois factory, which caused a fire in Arthur’s home in the Middle District of Georgia.


Answer:<br>
The Southern District of Illinois is not a proper venue under §1391 because no events giving rise to the claim took place there and Cleveland does not reside there under the venue statute.


Analysis:<br>
So, C is the right answer. The Southern District of Illinois is not a proper venue.
### Response
Explanation:<br>
The Southern District of Illinois cannot serve as a proper venue for Arthur's lawsuit against Cleveland Manufacturing Company under Section 1391(d) because Cleveland lacks sufficient contacts in this district to be considered as 'residing' there within the venue statute.

## Multiple choice mutation - Stage A
### System Instructions
You are an expert lawyer in the domain of U.S. civil procedure. You are given an introduction that describes a legal case, a question about that case and an answer to that question. First, you must identify what legal concept the question is talking about. Note that the given question contains some background on the details of the case. Provide a different background for your question and then ask a multiple choice question in the style of the original question, to illustrate that concept to legal law students. Your output should have the format: Concept:\\n<concept>\\n\\nBackground:\\n<background>\\n\\nQuestion:\\n<question>\\n\\nChoices\\n<choice_A>\\n<choice_B>\\n<choice_C>\\n<choice_D>\\n\\nCorrect Answer:\\n<correct_answer>.
### Example
Introduction:<br>
The venue provisions for cases involving corporate defendants are stated in Section 1391(b) of U.S. law. This section clarifies where a case may be brought; either the district where all defendants reside or where a significant portion of the events related to the claim occurred. However, defining where a corporation resides can be complex. Sections 28 U.S.C. §1391(c)(2) and (d) deal with this issue. It defines the residence of a corporation as any judicial district where it is under the court's personal jurisdiction concerning a particular civil action. If a state has more than one judicial district, a corporation is deemed to reside in any district within the state where it has sufficient contacts to be subjected to personal jurisdiction. If there is no such district, the corporation is considered to reside in the district with which it has the most significant contacts. An example is provided, using a fictional corporation and a legal dispute. Multiple potential district venues are considered, shedding light on the application of the provisions.

Question:<br>
Chu, a Californian, went skiing at Aspen, in Colorado, which has only one federal judicial district. After he discovered that he was somewhat out of shape, he bought an exercise machine from Jackson, a friend from Kansas with a two-week time share in Aspen, who had brought the machine with him to Aspen after buying it in Texas. The machine was made by Sweda- Trak, a Texas company that sells its products only in Texas, and with its only place of business in the Western District of Texas. Chu brought the machine back to Los Angeles, California, where he was injured using it. Chu brings a diversity action against Jackson and Sweda-Trak in federal court, alleging that Jackson was negligent in maintaining the machine, and Sweda-Trak in designing it. Venue in Chu’s action would be

Answer:<br>
proper in the Southern District of California under §1391(b)(2), because a substantial part of the events giving rise to the claim took place there.
### Response
Concept: <br>
Venue in Civil Procedure
      
Background:<br>
Sophia, a New York resident, wants to sue Eric, a Texas resident, claiming that he infringed on her copyright by using her work in his recent publication. Sophia has never been to Texas, but she knows that Eric frequently visits New York for business meetings. According to U.S. law, venue refers to the most appropriate location for a trial. It can be the district where the defendant resides or where the events related to the claim occurred.

Question:<br>
In which district would venue be proper if Sophia brings a copyright infringement lawsuit against Eric in federal court?

Choices:<br>
A) The Southern District of New York, where Sophia resides<br>
B) The Western District of Texas, where Eric resides<br>
C) The Eastern District of New York, where Eric frequently visits for business meetings<br>
D) The Northern District of Texas, where Sophia's work was originally created<br>

Correct Answer:<br>
A) The Southern District of New York, where Sophia resides

## Multiple choice mutation - Stage B
### System Instructions
You are an expert lawyer in the domain of U.S. civil procedure. You are given an introduction that describes legal concepts, one of these concepts, background of a legal case and a multiple choice question about that case. First, answer which answer is the correct one and then for each choice explain in one sentence why it is correct or wrong. The format of your answer should be: The correct answer is:\\n<your_answer>\\n\\nChoices:\\n\\n<choice_A>\\n <explanation_for_choice_A>\\n<choice_B>\\n <explanation_for_choice_B>\\n<choice_C>\\n <explanation_for_choice_C>\\n<choice_D>\\n <explanation_for_choice_D>\\n.
### Example
Introduction:<br>
The venue provisions for cases involving corporate defendants are stated in Section 1391(b) of U.S. law. This section clarifies where a case may be brought; either the district where all defendants reside or where a significant portion of the events related to the claim occurred. However, defining where a corporation resides can be complex. Sections 28 U.S.C. §1391(c)(2) and (d) deal with this issue. It defines the residence of a corporation as any judicial district where it is under the court's personal jurisdiction concerning a particular civil action. If a state has more than one judicial district, a corporation is deemed to reside in any district within the state where it has sufficient contacts to be subjected to personal jurisdiction. If there is no such district, the corporation is considered to reside in the district with which it has the most significant contacts. An example is provided, using a fictional corporation and a legal dispute. Multiple potential district venues are considered, shedding light on the application of the provisions.

Concept:<br>
Venue in Civil Procedure

Background:<br>
Mark, a resident of Florida, wants to file a personal injury lawsuit against GreenCo, a multinational corporation headquartered in Delaware, for injuries sustained due to a defective product. The product was purchased online and delivered to Mark's home in Florida. Mark believes the defective product was manufactured and designed at GreenCo's facility in Kansas.

Question:<br>
In which district would venue be proper if Mark brings a personal injury lawsuit against GreenCo in federal court?

Choices:<br>
A) The District of Delaware, where GreenCo is headquartered<br>
B) The Middle District of Florida, where Mark resides<br>
C) The District of Kansas, where the defective product was manufactured and designed<br>
D) The Southern District of Florida, where the product was delivered to Mark's home<br>

### Response
The correct answer is:<br>
B) The Middle District of Florida, where Mark resides

Choices:<br>
A) The District of Delaware, where GreenCo is headquartered<br>
Incorrect, because while GreenCo's headquarters are in Delaware, there doesn't seem to be any significant actions related to the claim that occurred there.<br>

B) The Middle District of Florida, where Mark resides<br>
Correct, because the product was delivered to Mark's home in Florida, causing the injury. Therefore, a significant portion of the events related to the claim occurred there.<br>

C) The District of Kansas, where the defective product was manufactured and designed<br>
Incorrect, although the product was manufactured and designed in Kansas, the significant harm occurred in Florida, making this district less relevant for this case.<br>

D) The Southern District of Florida, where the product was delivered to Mark's home<br>
Incorrect, because Mark does not reside in the Southern District of Florida and no specific information was given that suggest the Southern District as the proper venue.<br>