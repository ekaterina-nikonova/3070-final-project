## 2. Literature review

### 2.1. Academic sources

#### 2.1.1. Research on technology-assisted Japanese language learning 

**Automated Handwriting Recognition and Scoring for Japanese** achieved significant advances in a study by Nguyen et al. (2022). Researchers developed a system to automatically score handwritten descriptive answers from approximately 120,000 Japanese university entrance examinees during 2017–2018 trial examinations. The system employed ensemble deep neural networks (VGG16, MobileNet24, ResNet34, ResNeXt50, DenseNet121) trained on the ETL database, fine-tuned with only 0.5% labelled data from the actual exam dataset, and enhanced with n-gram language models. Character recognition accuracy exceeded 97%, and automatic scoring using BERT-based text classification achieved Quadratic Weighted Kappa (QWK) scores between 0.84 and 0.98, representing "almost perfect agreement" with human examiners.[[Nguyen, Nguyen]]

This research demonstrates the feasibility of automated Japanese handwriting recognition and assessment at scale. However, the system was designed for summative assessment of native speakers rather than formative feedback for learners. It does not provide the detailed, pedagogical feedback (stroke order errors, character confusions, comparative analysis with spoken responses) that would support skill development in foreign language learners.

**Automated Spoken Japanese Assessment** research has explored elicited imitation (EI) — a method where a learner repeats a spoken utterance after a teacher — combined with ASR for measuring oral proficiency. Matsushita and LeGare reported a correlation of 0.84 between human grading and ASR scoring of EI tests, though they noted the system required improvement for practical implementation. Recent advances (2025) at Doshisha University demonstrated that the Whisper ASR system produces transcriptions matching human raters with exceptional accuracy, even in non-ideal conditions with background noise. The system achieved fully automated scoring with a mean absolute percentage error (MAPE) of 7.230% using BERT-BiLSTM architecture with cosine similarity.[[Junaidi]][[Tsuchiya]][[Suzuki]][[McGuire]]

More specialised work on **phonemic transcription with accent markers** for Japanese-speaking assessments (2025) developed speech recognisers equipped with phonetic alphabet decoders using lattice fusion techniques. These systems were trained on the Corpus of Spontaneous Japanese (CSJ), which contains phonetic annotations of mispronunciations, hesitations, and accents. The research demonstrated that **multitask learning schemes** incorporating phonetic transcription prediction, text-token prediction, and fundamental frequency (f₀) pattern classification improved recognition accuracy for non-native speech.[[Kubo]]

#### 2.1.2. Theoretical foundations for the proposed system

Multimodal language learning research provides theoretical grounding for the proposed system. A study on robot-assisted Japanese vocabulary learning (2024) investigated whether multimodal presentation of referents (combining visual objects, images, and spoken words) enhances learning outcomes compared to unimodal presentation. Results suggested that integrated multimodal input facilitates deeper encoding and more robust memory formation.[[Wolfert]]

Research on Japanese pitch accent learning demonstrated that combining auditory input with visual pitch displays and hand gestures significantly improved perception accuracy compared to audio-only training. The study revealed that left-hand gestures (L-gestures), which activate the right hemisphere associated with pitch processing in novices, were particularly effective for beginners. These findings support the inclusion of multiple feedback modalities (visual, auditory, and potentially gestural cues) in pronunciation instruction.[[Hirata]]

Multimodal Corrective Feedback (MCF) research in German language education, which may be applicable to Japanese, demonstrated that combining verbal corrections with nonverbal cues, such as gaze, hand gestures, intonation changes, significantly enhanced learner attention and comprehension. Over 90% of students in the study emphasised the significant role of multimodal feedback in delivering corrections, attributing heightened attention and enjoyment to these interactions.

The findings suggest that feedback modality diversity enhances both cognitive processing and affective engagement.[[Guo]]

### 2.2. Critical evaluation of existing commercial solutions

A number of commercial solutions assist Japanese learners in language acquisition and assessment using multimodal features, but few of them integrate the comprehensive feature set of the proposed system.

**SALAD (Smart AI Language Assistant Daily)** represents a recent AI-driven approach targeting English-speaking learners of Japanese. The system integrates translation services (Kanji-Kana-Romaji), speech recognition, text-to-speech, vocabulary tracking, grammar explanations, and even song generation from learned vocabulary. A survey of 41 users revealed that 58.5% found the integrated learning features "extremely useful," with over 60% expressing confidence in the app's potential to improve their Japanese proficiency.[[Nihal]]

However, SALAD exhibits significant limitations relevant to the proposed system. First, it functions primarily as a **translation-centric tool** rather than an exercise generation system, maintaining users in a reactive learning mode rather than productive practice. The vocabulary tracking is passive, monitoring words encountered during translation rather than systematically building fluency with controlled vocabulary. Most critically, SALAD lacks any handwriting component and does not implement the dual-modality verification approach central to the proposed system. The song generation feature, while innovative for engagement, addresses entertainment rather than systematic skill consolidation.

**Duolingo's Japanese course** excels at delivering bite-sized lessons that encourage habit formation and steady engagement. Its writing-system tool for hiragana and katakana, which pairs tracing with sound matching, is well-designed, and story-based content offers contextual, engaging input. The course provides a clear progression from beginner to intermediate levels, and its large user base supports continual model refinement.
However, the course separates writing-system practice from vocabulary and grammar instruction and does not integrate handwriting with semantic or grammatical content. Speaking practice relies on generic speech recognition without Japanese-specific phonemic or pitch accent feedback, and there is no comparative analysis of handwritten versus spoken production. Vocabulary exposure is predetermined rather than learner-controlled, grammatical instruction is limited, i.e. rules are learned mainly by induction from examples, and feedback is typically binary rather than diagnostically detailed.[[Duolingo]]

**WaniKani** and **Bunpro** represent complementary tools focusing on kanji/vocabulary and grammar respectively. WaniKani employs a mnemonic-heavy, spaced repetition system (SRS) to teach 2,136 kanji and approximately 6,000 vocabulary items through a rigid 60-level progression. While effective for systematic kanji acquisition, WaniKani's inflexibility prevents learners from focusing on personally relevant vocabulary or skipping to appropriate levels based on existing knowledge. The vocabulary is selected specifically to reinforce kanji learning rather than for communicative utility. Bunpro addresses grammar systematically but remains separate from integrated practice activities.

Neither WaniKani nor Bunpro incorporates handwriting practice or pronunciation assessment. The assessment is unidirectional: learners type readings or meanings but do not produce handwritten characters or spoken responses. This constitutes a critical gap, as research demonstrates that handwriting practice correlates strongly with comprehension and retention.[[Otsuka - multi]]

**AI-Powered Tutoring Systems** like **iVoca**, **My SenpAI**, and **SakuraSpeak** provide conversational practice and pronunciation feedback through speech recognition. My SenpAI offers real-time pronunciation coaching with pitch accent training, visual displays of spoken input, and shadowing exercises. SakuraSpeak creates life-like conversations for practice in both English and Japanese. However, these systems focus exclusively on oral communication, neglecting the critical handwriting dimension that research identifies as essential for comprehensive literacy development.
[[Otsuka - unique]]
[[Otsuka - cognitive]]

The table below summarises the existing solutions and compares them to the proposed system:

| Application           | Handwriting Practice                             | Speech Recognition/Pronunciation             | Vocabulary Expansion                | Multimodal Feedback                                         | Internal Consistency Check                | Contextual Text Generation   |             Assessment Focus             |
|:----------------------|:-------------------------------------------------|:---------------------------------------------|:------------------------------------|:------------------------------------------------------------|:------------------------------------------|:-----------------------------|:----------------------------------------:|
| SALAD                 | No                                               | Yes - Text-to-speech output                  | Yes - Primary focus                 | Limited - separate modalities                               | No                                        | Translation-based, reactive  |           Translation accuracy           |
| Duolingo              | Removed (previously available)                   | Yes - Basic pronunciation                    | Yes - Constant                      | No                                                          | No                                        | Limited, rigid structures    |    Multiple-choice, typing, speaking     |
| WaniKani              | No - recognition only                            | No                                           | Yes - 6,000+ words across 60 levels | No                                                          | No                                        | Isolated vocabulary items    |     Kanji and vocabulary recognition     |
| Bunpro                | No                                               | No                                           | No - Grammar-focused                | No                                                          | No                                        | Example sentences            |       Grammar pattern recognition        |
| Rosetta Stone         | Limited - script tracing only                    | Yes - TruAccent technology                   | Yes - Structured progression        | Limited                                                     | No                                        | Image-based immersion        |     Comprehension and pronunciation      |
| My SenpAI             | No                                               | Yes - Pitch accent training, visual feedback | No - Conversational practice        | Yes - Pronunciation coaching                                | No                                        | Conversational dialogue      |        Spoken fluency and accent         |
| SakuraSpeak           | No                                               | Yes - Conversation simulation                | No - Practice-based                 | Limited                                                     | No                                        | AI-generated conversations   |        Conversational interaction        |
| **Proposed Solution** | **Yes - Full motor production + OCR assessment** | **Yes - ASR with pitch accent analysis**     | **No - Consolidation only**         | **Yes - Semantic, grammatical, handwriting, pronunciation** | **Yes - Cross-modal validation required** | **RAG-based coherent texts** | **Multimodal consolidation and fluency** |


This table clearly illustrates that no existing solution integrates the comprehensive feature set of the proposed system. The proposed solution is the only application that combines all mentioned features: full handwriting production with OCR assessment, comprehensive multimodal feedback across four dimensions, internal consistency validation between handwritten and spoken responses, strict adherence to the controlled vocabulary paradigm, and RAG-based contextual text generation. This feature integration represents a genuinely novel approach to Japanese language consolidation and assessment.

***

## References

Hung Tuan Nguyen, Cuong Tuan Nguyen, Haruki Oka, Tsunenori Ishioka, and Masaki Nakagawa. 2022. Handwriting Recognition and Automatic Scoring for Descriptive Answers in Japanese Language Tests. In Frontiers in Handwriting Recognition: 18th International Conference, ICFHR 2022, Hyderabad, India, December 4–7, 2022, Proceedings. Springer-Verlag, Berlin, Heidelberg, 274–284. https://doi.org/10.1007/978-3-031-21648-0_19

Junaidi, M. A. D., & Ratna, A. A. P. (2025). DEVELOPMENT AND OPTIMIZATION OF SIMPLE-O: AN AUTOMATED ESSAY SCORING SYSTEM FOR JAPANESE LANGUAGE BASED ON BERT, BILSTM, AND BIGRU. International Journal of Electrical, Computer, and Biomedical Engineering, 3(3), 579–597. https://doi.org/10.62146/ijecbe.v3i3.100

Tsuchiya, S. (2011). Elicited Imitation and Automated Speech Recognition: Evaluating Differences among Learners of Japanese.

Suzuki M. and Harada Y. (2005). Using Speech Recognition for an Automated Test of Spoken Japanese. In Proceedings of the 19th Pacific Asia Conference on Language, Information and Computation, pages 317–323, Taipei, Taiwan, R.O.C.. Institute of Linguistics, Academia Sinica.

McGuire M. (2025). Research News: Towards a Fully Automated Approach for Assessing English Proficiency (https://research.doshisha.ac.jp/news/news-detail-73/)

Guo, X. (2023). Multimodality in language education: implications of a multimodal affective perspective in foreign language teaching. Frontiers in Psychology, 14, 1283625.

Kubo, Yotaro & Sproat, Richard & Taguchi, Chihiro & Jones, Llion. (2025). Building Tailored Speech Recognizers for Japanese Speaking Assessment. 10.48550/arXiv.2509.20655. 

Wolfert, P., De Gersem, L., Janssens, R., & Belpaeme, T. (2024). Multi-modal language learning: explorations on learning Japanese vocabulary. In D. Grollman & E. Broadbent (Eds.), COMPANION OF THE 2024 ACM/IEEE INTERNATIONAL CONFERENCE ON HUMAN-ROBOT INTERACTION, HRI 2024 COMPANION (pp. 1129–1133). https://doi.org/10.1145/3610978.3640685

Hirata, Y., Friedman, E., Kaicher, C., & Kelly, S. D. (2024). Multimodal training on L2 Japanese pitch accent: learning outcomes, neural correlates and subjective assessments. Language and Cognition, 16(4), 1718–1755. doi:10.1017/langcog.2024.24

Nihal, R.A., Tran, D.H., Lin, Z., Xu, Y., Liu, H., An, Z., & Kyou, M. (2024). SALAD: Smart AI Language Assistant Daily. ArXiv, abs/2402.07431.

Strong E., Denter C., Chan A. (2023) A new tool for learning to read Japanese on Duolingo. https://blog.duolingo.com/learning-to-read-japanese-characters/

Otsuka, S., & Murai, T. (2020). The multidimensionality of Japanese kanji abilities. Scientific Reports, 10, 3039. https://doi.org/10.1038/s41598-020-59852-0

Otsuka, S., & Murai, T. (2023). The unique contribution of handwriting accuracy to literacy skills in Japanese adolescents. Reading and writing, 1–26. Advance online publication. https://doi.org/10.1007/s11145-023-10433-3

Otsuka, S., & Murai, T. (2021). Cognitive underpinnings of multidimensional Japanese literacy and its impact on higher-level language skills. Scientific reports, 11(1), 2190. https://doi.org/10.1038/s41598-021-81909-x
