University of London, 2025

BSc CSc | CM3070 Final Project

Ekaterina Nikonova

***

# Preliminary report

# AI-Powered Japanese Language Exercise System with Multimodal Feedback

***

## 1. Project concept

### 1.1. Primary concept

The proposed **AI-Powered Japanese Language Exercise System with Multimodal Feedback** represents a comprehensive solution that addresses a critical gap in current Japanese language acquisition tools: the integration of multimodal skill assessment (handwriting and speaking) with controlled vocabulary exposure. Unlike conventional language learning platforms that continuously introduce new vocabulary and grammar, this system operates as a consolidation and fluency-building device that reinforces existing knowledge through contextually coherent exercises.

The project is based on the template *CM3020 Artificial Intelligence. Project Idea 1: Orchestrating AI models to achieve a goal* and combines three types of AI models: Large Language Models (LLMs), Automatic Speech Recognition (ASR), and Optical Character Recognition (OCR) to generate training content, interpret the learner's responses, and provide comprehensive, actionable feedback.

Design choices for the application prioritise user control and minimal data exposure. The system defaults to local processing for sensitive signals (audio and handwriting) where device capabilities permit, and uploads occur only with explicit, informed user actions. The application explains in detail what information is sent to third-party services. Since user's inputs may content sensitive data even after converting into text, responses are not stored anywhere except the user's local system.

The architecture emphasises lightweight, modular components so deployments can scale from offline desktop or mobile applications to cloud-assisted services. Core assessment routines and lightweight models can run locally on consumer-grade devices to reduce barriers for individual learners.

### 1.2. Motivation and importance of multimodal assessment

The motivation for this system emerges from well-documented challenges in Japanese language acquisition, particularly for learners from non-kanji linguistic backgrounds. Research demonstrates that handwriting practice significantly enhances literacy acquisition and retention in Japanese. Studies reveal that the coupling of motor action and perception during handwriting facilitates kanji literacy, while reduced writing ability is observed among Japanese adults who decrease handwriting practice. For non-native learners, this motor-perceptual integration is even more critical, as handwriting reinforces character recognition, stroke order awareness, and overall comprehension.[Otsuka, 2020]

The integration of handwriting and speaking assessment addresses fundamental characteristics of Japanese language competence. Unlike alphabetic languages where a single writing system maps to phonology, Japanese employs three distinct scripts (hiragana, katakana, kanji) with complex orthography-phonology relationships. Learners must, therefore, develop separate but interconnected competencies covering all three groups of characters, including visual recognition, motor production, and auditory comprehension.

Moreover, the cultural context of Japanese language use emphasises orthographic precision and handwriting aesthetics as markers of education and respect. While digital communication has reduced handwriting frequency in daily life, the ability to produce clear, correctly formed characters remains valued in formal and professional contexts. Foreign language learners benefit from handwriting practice not only for cognitive reasons but also for cultural integration and professional credibility.

### 1.3. Project differentiation

Unlike systems that continuously introduce new material, this system operates exclusively within the learner's established vocabulary inventory. This addresses a fundamental pedagogical challenge: learners often accumulate superficial knowledge of many words without achieving fluency or automaticity. By constraining text generation to familiar elements, the system creates a safe practice space where cognitive resources are focused on integration and production rather than decoding.

The system's explicit framing as an "exercise machine" rather than a "teaching tool" addresses learner motivation and sustainable practice. By analogy to physical fitness equipment, it provides structured, repeatable practice that consolidates existing skills without the cognitive overwhelm of constant novelty. This positions the system as complementary to other resources, such as textbooks, classes, or immersion experiences, that introduce new material, serving a distinct and necessary role in the learning ecosystem.

Existing automated assessment systems typically provide holistic scores or error counts without detailed pedagogical guidance. The proposed system delivers dimension-specific feedback addressing semantic appropriateness, grammatical accuracy, handwriting quality, and pronunciation — each with concrete suggestions for improvement. This aligns with research showing that multimodal corrective feedback enhances both comprehension and motivation.

### 1.4. Positioning in the learning ecosystem

The **AI-Powered Japanese Language Exercise System with Multimodal Feedback** represents a necessary evolution in computer-assisted language learning for Japanese. By integrating handwriting and speech production within a controlled-vocabulary framework, employing RAG for coherent text generation, and providing comprehensive multimodal feedback, the system addresses critical gaps in current offerings. It serves neither as a replacement for systematic instruction nor as a direct competitor to vocabulary-building tools, but rather as an essential complement that facilitates the transition from recognition to production, from isolated knowledge to integrated fluency. As research consistently demonstrates the importance of multimodal practice and the unique role of handwriting in Japanese literacy, this system provides an evidence-based, scalable solution for learners seeking to consolidate and automatise their Japanese language skills.

## 2. Literature review

### 2.1. Academic sources

#### 2.1.1. Research on technology-assisted Japanese language learning 

Automated handwriting recognition for the Japanese language achieved significant advances. Researchers developed a system to automatically score handwritten descriptive answers from approximately 120 thousand Japanese university entrance examinees during 2017–2018 trial examinations. The system employed ensemble deep neural networks trained on the ETL database, fine-tuned with only 0.5% labelled data from the actual exam dataset, and enhanced with n-gram language models. Character recognition accuracy exceeded 97%, and automatic scoring using BERT-based text classification achieved Quadratic Weighted Kappa (QWK) scores between 0.84 and 0.98, representing "almost perfect agreement" with human examiners.[Nguyen, 2022]

This research demonstrates the feasibility of automated Japanese handwriting recognition and assessment at scale. However, the system was designed for summative assessment of native speakers rather than formative feedback for learners. It does not provide the detailed pedagogical feedback on stroke order errors, character confusions, or comparative analysis with spoken responses that would support skill development in foreign language learners.

Automated spoken Japanese assessment research has explored elicited imitation (EI) — a method where a learner repeats a spoken utterance after a teacher — combined with ASR for measuring oral proficiency. The reported correlation between human grading and ASR scoring of EI tests reaches 0.84, though it is noted that the system required improvement for practical implementation. Recent advances demonstrated that the Whisper ASR system produces transcriptions matching human raters with exceptional accuracy, even in non-ideal conditions with background noise. The system achieved fully automated scoring with a mean absolute percentage error (MAPE) of 7.230% using BERT-BiLSTM architecture with cosine similarity.[Junaidi, 2025; Tsuchiya, 2011; Suzuki, 2005; McGuire, 2025]

More specialised work on phonemic transcription with accent markers for Japanese-speaking assessments developed speech recognisers equipped with phonetic alphabet decoders using lattice fusion techniques. These systems were trained on the Corpus of Spontaneous Japanese (CSJ), which contains phonetic annotations of mispronunciations, hesitations, and accents. The research demonstrated that multitask learning schemes incorporating phonetic transcription prediction, text-token prediction, and fundamental frequency (f₀) pattern classification improved recognition accuracy for non-native speech.[Kubo, 2025]

#### 2.1.2. Theoretical foundations for the proposed system

Multimodal language learning research provides theoretical grounding for the proposed system. A study on robot-assisted Japanese vocabulary learning investigated whether multimodal presentation of referents combining visual objects, images, and spoken words enhances learning outcomes compared to unimodal presentation. Results suggested that integrated multimodal input facilitates deeper encoding and more robust memory formation.[Wolfert, 2024]

Research on Japanese pitch accent learning demonstrated that combining auditory input with visual pitch displays and hand gestures significantly improved perception accuracy compared to audio-only training. The study revealed that left-hand gestures (L-gestures), which activate the right hemisphere associated with pitch processing in novices, were particularly effective for beginners. These findings support the inclusion of multiple feedback modalities — visual, auditory, and potentially gestural cues — in pronunciation instruction.[Hirata, 2024]

Multimodal Corrective Feedback (MCF) research in German language education, which may be applicable to Japanese, demonstrated that combining verbal corrections with nonverbal cues, such as gaze, hand gestures, and intonation changes, significantly enhanced learner attention and comprehension. Over 90% of students in the study emphasised the significant role of multimodal feedback in delivering corrections, attributing heightened attention and enjoyment to these interactions.

The findings suggest that feedback modality diversity enhances both cognitive processing and affective engagement.[Guo, 2023]

### 2.2. Critical evaluation of existing commercial solutions

A number of commercial solutions assist Japanese learners in language acquisition and assessment using multimodal features, but few of them integrate the comprehensive feature set of the proposed system.

**SALAD (Smart AI Language Assistant Daily)** represents a recent AI-driven approach targeting English-speaking learners of Japanese. The system integrates translation services, speech recognition, text-to-speech, vocabulary tracking, grammar explanations, and even song generation from learned vocabulary. A survey of 41 users revealed that 58.5% found the integrated learning features "extremely useful", with over 60% expressing confidence in the application's potential to improve their Japanese proficiency.[Nihal, 2024]

However, SALAD exhibits significant limitations relevant to the proposed system. First, it functions primarily as a translation-centric tool rather than an exercise generation system, maintaining users in a reactive learning mode rather than productive practice. The vocabulary tracking is passive, monitoring words encountered during translation rather than systematically building fluency with controlled vocabulary. Most critically, SALAD lacks any handwriting component and does not implement the dual-modality verification approach central to the proposed system. The song generation feature, while innovative for engagement, addresses entertainment rather than systematic skill consolidation.

**Duolingo's Japanese course** excels at delivering bite-sized lessons that encourage habit formation and steady engagement. Its writing-system tool for hiragana and katakana, which pairs tracing with sound matching, is well-designed, and story-based content offers contextual, engaging input. The course provides a clear progression from beginner to intermediate levels, and its large user base supports continual model refinement.

However, the course separates writing-system practice from vocabulary and grammar instruction and does not integrate handwriting with semantic or grammatical content. Speaking practice relies on generic speech recognition without Japanese-specific phonemic or pitch accent feedback, and there is no comparative analysis of handwritten versus spoken production. Vocabulary exposure is predetermined rather than learner-controlled, grammatical instruction is limited, i.e. rules are learned mainly by induction from examples, and feedback is typically binary rather than diagnostically detailed.[Strong, 2023]

**WaniKani** and **Bunpro** represent complementary tools focusing on kanji/vocabulary and grammar respectively. WaniKani employs a mnemonic-heavy, spaced repetition system to teach over 2000 kanji characters and approximately 6000 vocabulary items through a rigid 60-level progression. While effective for systematic kanji acquisition, WaniKani's inflexibility prevents learners from focusing on personally relevant vocabulary or skipping to appropriate levels based on existing knowledge. The vocabulary is selected specifically to reinforce kanji learning rather than for communicative utility. Bunpro addresses grammar systematically but remains separate from integrated practice activities.

Neither WaniKani nor Bunpro incorporates handwriting practice or pronunciation assessment. The assessment is unidirectional: learners type readings or meanings but do not produce handwritten characters or spoken responses. This constitutes a critical gap, since research demonstrates that handwriting practice correlates strongly with comprehension and retention.[Otsuka, 2020]

**Rosetta Stone** provides immersive, contextual learning with real speech recognition feedback. It employs immersive methodology and assesses the quality of pronunciation using a proprietary TruAccent technology. Yet like other platforms, the service treats speaking and writing as separate domains, and its speech recognition doesn't address Japanese pitch accent, which is a fundamental feature that distinguishes word meanings.

AI-powered tutoring systems like **My SenpAI**, and **SakuraSpeak** provide conversational practice and pronunciation feedback through speech recognition. My SenpAI offers real-time pronunciation coaching with pitch accent training, visual displays of spoken input, and shadowing exercises. SakuraSpeak creates life-like conversations for practice in both English and Japanese. However, these systems focus exclusively on oral communication, neglecting the critical handwriting dimension that research identifies as essential for comprehensive literacy development.[Otsuka, 2023; Otsuka, 2021]

The table below summarises features of the existing solutions and compares them to the proposed system:

| Application           | Handwriting Practice                             | Speech Recognition/Pronunciation             | Vocabulary Expansion                | Multimodal Feedback                                         | Internal Consistency Check                | Contextual Text Generation   |             Assessment Focus             |
|:----------------------|:-------------------------------------------------|:---------------------------------------------|:------------------------------------|:------------------------------------------------------------|:------------------------------------------|:-----------------------------|:----------------------------------------:|
| SALAD                 | No                                               | Yes - Text-to-speech output                  | Yes - Primary focus                 | Limited - separate modalities                               | No                                        | Translation-based, reactive  |           Translation accuracy           |
| Duolingo              | Limited (separate characters)                    | Yes - Basic pronunciation                    | Yes - Constant                      | No                                                          | No                                        | Limited, rigid structures    |    Multiple-choice, typing, speaking     |
| WaniKani              | No - recognition only                            | No                                           | Yes - 6,000+ words across 60 levels | No                                                          | No                                        | Isolated vocabulary items    |     Kanji and vocabulary recognition     |
| Bunpro                | No                                               | No                                           | No - Grammar-focused                | No                                                          | No                                        | Example sentences            |       Grammar pattern recognition        |
| Rosetta Stone         | Limited - script tracing only                    | Yes - TruAccent technology                   | Yes - Structured progression        | Limited                                                     | No                                        | Image-based immersion        |     Comprehension and pronunciation      |
| My SenpAI             | No                                               | Yes - Pitch accent training, visual feedback | No - Conversational practice        | Yes - Pronunciation coaching                                | No                                        | Conversational dialogue      |        Spoken fluency and accent         |
| SakuraSpeak           | No                                               | Yes - Conversation simulation                | No - Practice-based                 | Limited                                                     | No                                        | AI-generated conversations   |        Conversational interaction        |
| **Proposed Solution** | **Yes - Full motor production + OCR assessment** | **Yes - ASR with pitch accent analysis**     | **No - Consolidation only**         | **Yes - Semantic, grammatical, handwriting, pronunciation** | **Yes - Cross-modal validation required** | **RAG-based coherent texts** | **Multimodal consolidation and fluency** |


This table clearly illustrates that no existing solution integrates the comprehensive feature set of the proposed system. The proposed solution is the only application that combines all mentioned features: full handwriting production with OCR assessment, comprehensive multimodal feedback across four dimensions, internal consistency validation between handwritten and spoken responses, strict adherence to the controlled vocabulary paradigm, and RAG-based contextual text generation. This feature integration represents a genuinely novel approach to Japanese language consolidation and assessment.

## 3. Design and implementation

### 3.1 Project overview

The **AI-Powered Japanese Language Exercise System with Multimodal Feedback** is based on *Template: CM3020 Artificial Intelligence. Project Idea 1: Orchestrating AI models to achieve a goal*. It is an intelligent tutoring system designed to support Japanese language learners in consolidating and improving their skills in four key competencies: written accuracy, pronunciation, handwriting, and comprehension. Rather than introducing new vocabulary or grammatical structures, the system functions as a retention and fluency training aid that operates on learner-curated vocabulary obtained from other resources.

The system employs orchestration of pre-trained models including semantic vector embeddings, Large Language Models (LLMs), Automatic Speech Recognition (ASR), and Optical Character Recognition (OCR). It provides actionable feedback across handwritten responses, spoken answers, and comprehension activities. With the help of Retrieval-Augmented Generation (RAG), the system generates contextually coherent exercises utilising the learner's established lexical and grammatical inventory.

### 3.2. Domain and users

The system focuses on the acquisition of Japanese as a foreign language. Its pedagogical aim is to enhance retention and communicative proficiency rather than facilitate initial vocabulary acquisition. The project's instructional goals prioritise fluency through repeated practice, accuracy via multimodal feedback, and explicit handwriting skill development that complements typing-based practice acquired through other types of instruction.

Primary users are beginner-to-intermediate Japanese learners who already possess foundational knowledge and need structured practice to consolidate material. The system also targets self-directed adult learners and students using classroom materials who want supplementary practice outside formal instruction. Secondary users include language teachers who will use the system for formative assessment and curriculum designers or institutions seeking to augment courseware with aligned practice tools.

Learners require immediate, actionable feedback across spoken, typed, and handwritten responses, and exercises must remain within their curated vocabulary to avoid introducing unfamiliar items prematurely. They also benefit from varied, engaging exercise contexts. Practical constraints include sensitivity to feedback latency, high variability in handwriting quality, differing levels of technical literacy among adult users, and potentially limited internet bandwidth in some deployment contexts.

### 3.3. System architecture

The system workflow consists of several interconnected components. First, a large language model generates sentences from a pre-defined vocabulary list representing the learner's current knowledge base. These sentences are embedded in a vector database to enable semantic retrieval. This operation is performed once for target vocabulary.

When a learner specifies a topic of interest, the system employs Retrieval-Augmented Generation (RAG) to construct coherent texts using only familiar linguistic elements. Simultaneously, comprehension questions are generated to assess understanding.

The system's defining characteristic emerges in its multimodal assessment approach. Learners respond to questions both in handwritten form and through spoken input. The OCR model converts handwritten responses to text, while the ASR model processes spoken answers. The dual-modality requirement creates a unique validation mechanism: ideally, both responses should be identical, providing an internal consistency check. Both responses are then evaluated by an LLM for semantic correctness and relevance to the question.

The feedback is comprehensive, addressing four dimensions:

- **semantic correctness** — "Does the answer address the question appropriately?"
- **grammatical accuracy** — "Is the language used in the response grammatically correct?" 
- **handwriting quality** — ""Are there character recognition errors indicating poor stroke order or form?" 
- **pronunciation accuracy** — "Does the spoken output match phonemic expectations?"

The system's architecture is illustrated below:

<img src="architecture.png" alt="image" style="zoom:150%;" />

### 3.4. Design justification

#### 3.4.1 Pedagogical

The design emphasises comprehensible input: exercises are constrained to the learner's existing vocabulary so that all content remains understandable. Multimodal speech production and explicit handwriting practice are included to improve pronunciation, form-meaning connections, and motor-visual skills that typing or speaking alone cannot address.

#### 3.4.2 Technical

RAG over a vector database is an efficient way to limit the context passed to the text-generating LLM, to ground the outputs in the learner's vocabulary, preventing vocabulary leakage.

The solution orchestrates specialised models (LLM, ASR, OCR, embeddings) so each modality is handled by the best-suited model. Even though modern state-of-the-art LLMs are capable of both speech recognition and handwriting OCR, they cannot be hosted on consumer-grade hardware. Moreover, the cost of using a commercial solution for any other inputs than text may be prohibitive for many users.

The system adds multimodal consistency checks to separate input-processing errors from linguistic errors and employs LLMs as rubric-based judges for structured, consistent evaluation.

#### 3.4.3 Practical

Requiring a learner-specified vocabulary list ensures curriculum alignment and makes the system a supplement to instruction rather than a replacement. Feedback is layered, which makes diagnostics actionable, and dual input increases the learner's engagement.

### 3.5. Work plan

The image below shows a timeline of the project's development:

<img src="work-plan.png" alt="A timeline showing the work plan" />

#### 3.5.1. Completed stages

After selecting the template, the work started with compiling the **vocabulary dataset**. Because of copyright restrictions linked with most existing Japanese language materials available commercially, it was decided to use generated sentences to populate the vector store for further retrieval. After many unsuccessful attempts to generate sentences from a pre-defined list of words and phrases using a local model, the project switched to using a remote model. As a result, an algorithm was created to repeatedly invoke the Perplexity Sonar model via an API with a subsection of the vocabulary for each entry. The resulting dataset consists of 7833 unique Japanese sentences, making sure that each entry in the vocabulary list is represented at least once.

The **RAG** pipeline was created simultaneously with the dataset preparation, and its word-retrieval component was used during the sentence generation stage.

A number of **OCR and ASR models** were tested and selected for use in the prototype. *Manga OCR* was selected for its ability to process multiline text, whereas alternatives such as *PaddleOCR* were considered but rejected due to their poor performance on images of non-standard sizes. *wav2vec2* was selected as an ASR model for its high accuracy and convenience of deployment.

Simultaneously, the **Background research and positioning** were carried out to identify potential use cases and existing solutions, as a preparation for the **Project pitch** delivery.

After the **RAG** stage was completed, the work started on combining the elements of the project, previously developed separately, into a single **Prototype** application. This required the utilisation of the *uv* library for managing Python environments, since different components of the application required different versions of Python and data-processing libraries, such as *numpy*.

The **Content generation (local)** stage assumed generating text and comprehension questions using a local LLM model, but the experiments with small models with 270M-4B parameters showed that the quality of the generated text was not satisfactory. Therefore, the work proceeded with the Perplexity Sonar model via API.

The present **Preliminary report** describes the project and concludes the first phase of the project's development.

#### 3.5.2. Planned stages

Further work will involve ensuring **Vocabulary constraints** either via prompt engineering, template-based approach, or result filtering via a feedback loop. **Alternatives to OCR and ASR** models will be evaluated to verify if increased accuracy is possible with a different model. Another attempt to use a **local LLM** for text generation will be made, with the fallback option of using the remote model, like in the presented prototype, if the quality or the performance of local models are not satisfactory.

In parallel with the aforementioned improvements, work on the **Web or mobile UI** will be carried out, as the changes in the application's components will not require alterations in the user workflow or the data flow in the system. The evaluation and testing will be performed iteratively at every stage, ensuring there is no regression in the system's functionality.

The **Final report** will be prepared in parallel with the development of the application.

### 3.6. Evaluation plan

This testing and evaluation plan defines measurable objectives, the methods to verify them, anticipated failure modes with mitigations, and the criteria that constitute project success.

#### 3.6.1. Evaluation goals and failure modes

The evaluation goals focus on measurable performance across modalities, exercise and feedback quality, learning outcomes, usability, and reliability, specifically:

- response processing accuracy aims for average ASR and OCR confidence consistent with reliable conversion: learner inputs must be correctly converted with >85% confidence on average;
- exercise generation should maintain semantic coherence and strict vocabulary compliance;
- feedback should correctly identify errors at a high rate;
- learners should show measurable improvements in retention after sustained use;
- usability targets include high task completion and user satisfaction;
- system uptime and latency targets must meet operational expectations.

Failure modes must be identified for each critical component and paired with mitigations. For example, persistent ASR errors due to accent can be mitigated by switching or retraining ASR models and providing manual input fallbacks. OCR failures from handwriting variability can be addressed with preprocessing, targeted training, and manual correction options using a feedback loop. LLM generation or evaluation failures that violate vocabulary constraints must be mitigated by strict post-generation filtering and human curation for edge cases.

#### 3.6.2. Efficiency of vocabulary embedding and retrieval

The efficiency of the vocabulary retrieval must be tested using a variety of approaches:
- embedding words and phrases with and without transcriptions;
- retrieving words and phrases by the lexeme versus the translation;
- evaluating RAG performance with the retrieval of raw vocabulary rather than sentences.

In all these cases, the retrieval efficiency must be measured against the expected performance of the system.

#### 3.6.3. Performance evaluation

Latency is a critical metric for maintaining the user's engagement. The system must be evaluated to determine the optimal latency targets for each component. Currently, the OCR and ASR are performed consecutively and together show the combined latency over 60 seconds, which is too high for real-time feedback. Afterwards, the usage of a remote LLM service introduces additional latency.

The performance evaluation and improvement plan includes
- caching of content generation results;
- asynchronous pipelines to reduce the waiting time until the first feedback;
- alternative OCR and ASR models for real-time feedback.

Project completion criteria include comprehensive documentation, reproducible deployment, and strong code quality and test coverage.

## 4. Prototype description and evaluation

### 4.1. Technical implementation and feasibility demonstration

The prototype demonstrates the technical feasibility of the proposed system through the integration of several key components: **retrieval-augmented generation (RAG)** for vocabulary-constrained content synthesis, **multimodal input processing** combining Optical Character Recognition (OCR) and Automatic Speech Recognition (ASR), and **LLM-based assessment** for evaluating linguistic correctness across written and spoken modalities.

#### 4.1.1 Vocabulary-controlled content generation via RAG

The system employs a vector store populated with A1-level Japanese sentences containing target vocabulary entries and syntactic patterns. When a learner specifies a topic, semantically related vocabulary is retrieved from this repository, and a prompt is constructed that instructs the LLM to generate text and comprehension questions using only the retrieved items. This approach addresses a critical pedagogical requirement: ensuring that exercise content remains within the learner's proficiency boundaries, thereby minimising the cognitive load. Moreover, this reduces the likelihood of introducing new lexical items without explicit pedagogical intent.

The prototype invokes Perplexity's Sonar model via API.

#### 4.1.2 Multimodal answer capture and conversion

The prototype successfully captures learner responses in two modalities:

- **Handwritten responses** are photographed and processed through OCR using the MangaOCR model to extract text.
- **Spoken responses** are recorded as audio files and processed through ASR using a fine-tuned variant of Facebook's XLSR-Wav2Vec2 model to produce transcriptions.

This dual-modality approach reflects the reality of language learning assessment, where both written and oral production are important measures of proficiency. The separation of these modalities in feedback generation allows the system to identify whether errors stem from orthographic or phonetic issues or deeper grammatical misunderstandings.

#### 4.1.3 LLM-based comparative assessment

The two transcriptions are compared, and the assessment model verifies that they have identical content, to then provide detailed feedback along four dimensions: semantic, grammatical, orthographic, and phonetic.

The image below shows the state of the application after the submission of responses for two questions. It demonstrates both the user input in the form of file paths to the handwritten answer image and audio recording, and the response from the LLM evaluator containing multi-modal feedback.

<img src="cli-feedback.png"  alt="App running in the terminal"/>

This multidimensional feedback mechanism corresponds to the principles of formative assessment in language learning, where immediate, specific feedback on multiple linguistic dimensions supports skill development [Black & William, 1998; Nicol & Macfarlane, 2006].

### 4.2. Evaluation of prototype performance and identified limitations

#### 4.2.1 Breaking vocabulary constraints

The generated text and accompanying questions are not consistently restricted to the vocabulary entries fetched from the vector store. Analysis of the prototype's outputs reveals instances where words appear that are not explicitly present in the learner's vocabulary list, particularly in the form of kanji or morphological variants.

The current implementation relies on semantic retrieval from the vector store to construct the prompt but does not enforce hard constraints on the LLM's output vocabulary. The model may generate plausible alternatives or morphological variations, such as verb conjugations or particles, that were not explicitly retrieved, treating these as semantically equivalent to the restricted vocabulary.

Plain validation via vocabulary look-up is challenging in the Japanese language, because Japanese text lacks whitespace delimiters, making detecting unknown words written in hiragana substantially more difficult. Unlike English or even kanji-heavy text, pure hiragana sequences provide no intrinsic morphological boundaries.

Instead, after generation the output can be processed with a comprehensive kanji dictionary, such as the KANJIDIC2 database used in NLP systems, to identify any kanji that are not in the learner's approved list. These characters can be flagged and the generation request rejected, requiring the LLM to regenerate the content using only approved kanji. Alternatively, a template-based lexically constrained text generation approach [Iso et al., 2022] can be used.

Another potential solution is to leverage Japanese morphological analysers such as MeCab (with an appropriate dictionary like NEologd) or JUMAN++ which can segment the hiragana text and identify word boundaries. The identified words can then be checked against the approved vocabulary list. However, these tools have known limitations: they may struggle with rare words or non-standard usage common in learner-produced or AI-generated text.

#### 4.2.2 Dependency on external LLM services

The prototype relies on the Perplexity Sonar model accessed via API. This introduces latency, dependency on external service availability, and ongoing API costs, limiting the system's applicability in offline or resource-constrained educational settings.

Recent developments in open-source, small-scale LLMs optimised for Japanese [SiliconFlow, 2025] make local deployment increasingly viable. However, our initial findings indicate that while smaller models can produce grammatically acceptable Japanese, they often:

- struggle with fine-grained vocabulary control when prompted with a vocabulary list that exceeds their context window,
- occasionally introduce hallucinations or semantically incoherent content,
- produce less nuanced feedback in comparative assessment tasks,
- show the reduced ability to maintain consistency across multi-turn interactions.

However, incremental improvements may be achievable through:
- fine-tuning on a corpus of vocabulary-restricted Japanese language exercises to improve adherence to vocabulary constraints,
- prompt engineering and inclusion of few-shot examples in the prompt,
- ensemble or cascading approaches, using a smaller local model for initial text generation, with verification and refinement via a larger external model only when quality thresholds are not met.

The field of efficient Japanese LLMs is rapidly evolving. Continued monitoring of model releases and performance benchmarks is essential to identify the optimal trade-off point between model size, quality, and latency for this educational application.

#### 4.2.3 OCR and ASR precision

While the prototype successfully demonstrates multimodal input processing, both OCR and ASR introduce errors that can cascade through the system. Similar-looking characters may be misidentified, leading to incorrect transcriptions of learner responses. Additionally, ASR systems are prone to homophony and context-dependency, leading to incorrect transcriptions of speech.

To assess the system performance, comparative evaluation of open-source OCR (Tesseract with Japanese language packs, PaddleOCR, EasyOCR) and ASR systems (Whisper, Faster-Whisper, Noto Speech Recognition) must be conducted against the current implementations.

Domain-specific adaptation via fine-tuning on a corpus of A1-level learner handwriting and speech can reduce the impact of these errors. However, this approach requires a large corpus of labelled handwritten and spoken data produced by beginner-level learners, which may not be available.

Instead, confidence-based filtering can be used to detect and reject incorrect transcriptions, where the system can request user confirmation or re-recording when the confidence of the generated output falls below a threshold.

#### 4.2.4 User interface and workflow

The prototype uses a command-line interface (CLI) that requires that the learners:
1. Manually photograph handwritten answers (producing PNG files)
2. Separately record audio responses (producing WAV files)
3. Save both files locally
4. Manually input file paths into the CLI
5. Ensure audio is encoded in the correct format (16-bit, 16 kHz WAV)

This workflow introduces friction and is not representative of how learners would naturally interact with the system in an educational setting.

The following improvements can be made to enhance usability:

1. A web-based or native mobile application could provide:
   - *real-time handwriting capture* using a tablet or touchscreen (with fallback to photograph upload),
   - *one-tap or -click audio recording* with automatic format conversion,
   - *preview and confirmation* before submission, reducing errors and requests for re-recording.

2. Automatic management of file encoding, format conversion, and clean-up can eliminate the need for user involvement in technical details.

3. Initially presented with a simple "Record" button, advanced learners or educators could access settings for fine-tuning quality parameters, such as noise suppression.

### 4.3. Summary

The prototype successfully demonstrates the feasibility of an integrated system for vocabulary-constrained exercise generation and multimodal assessment. However, several opportunities for improvement are evident:

1. Implementing a feedback loop with the detection of lexical units not present in the vocabulary can reduce constraint leakage while maintaining generation fluency.

2. Continued monitoring of small-scale, Japanese-specialised LLMs, combined with prompt engineering techniques, offers a path toward offline deployment without significant quality compromise.

3. Confidence-based filtering can improve transcription accuracy while reducing false positives in error detection.

4. A GUI that handles capture, format conversion, and file management will significantly improve usability for real-world deployment.

These improvements may contribute to positioning the system as a technically sound and pedagogically grounded contribution to the field.

***

## References

Black, P., & Wiliam, D. (1998). Assessment and classroom learning. *Assessment in Education: Principles, Policy & Practice*, 5(1), 7–74.

Guo, X. (2023). Multimodality in language education: implications of a multimodal affective perspective in foreign language teaching. Frontiers in Psychology, 14, 1283625.

Hirata, Y., Friedman, E., Kaicher, C., & Kelly, S. D. (2024). Multimodal training on L2 Japanese pitch accent: learning outcomes, neural correlates and subjective assessments. Language and Cognition, 16(4), 1718–1755. doi:10.1017/langcog.2024.24

Hung Tuan Nguyen, Cuong Tuan Nguyen, Haruki Oka, Tsunenori Ishioka, and Masaki Nakagawa. (2022). Handwriting Recognition and Automatic Scoring for Descriptive Answers in Japanese Language Tests. In Frontiers in Handwriting Recognition: 18th International Conference, ICFHR 2022, Hyderabad, India, December 4–7, 2022, Proceedings. Springer-Verlag, Berlin, Heidelberg, 274–284. https://doi.org/10.1007/978-3-031-21648-0_19

Iso, H., et al. (2022). A simple recipe for lexically constrained text generation. *Transactions of the Association for Computational Linguistics*, 10, 815–832.

Junaidi, M. A. D., & Ratna, A. A. P. (2025). DEVELOPMENT AND OPTIMIZATION OF SIMPLE-O: AN AUTOMATED ESSAY SCORING SYSTEM FOR JAPANESE LANGUAGE BASED ON BERT, BILSTM, AND BIGRU. International Journal of Electrical, Computer, and Biomedical Engineering, 3(3), 579–597. https://doi.org/10.62146/ijecbe.v3i3.100

Kubo, Yotaro & Sproat, Richard & Taguchi, Chihiro & Jones, Llion. (2025). Building Tailored Speech Recognizers for Japanese Speaking Assessment. 10.48550/arXiv.2509.20655. 

McGuire M. (2025). Research News: Towards a Fully Automated Approach for Assessing English Proficiency (https://research.doshisha.ac.jp/news/news-detail-73/)

Nihal, R.A., Tran, D.H., Lin, Z., Xu, Y., Liu, H., An, Z., & Kyou, M. (2024). SALAD: Smart AI Language Assistant Daily. ArXiv, abs/2402.07431.

Nicol, D., & Macfarlane-Dick, D. (2006). Formative assessment and self-regulated learning: A model and seven principles of good feedback practice. *Studies in Higher Education*, 31(2), 199–218.

Otsuka, S., & Murai, T. (2020). The multidimensionality of Japanese kanji abilities. Scientific Reports, 10, 3039. https://doi.org/10.1038/s41598-020-59852-0

Otsuka, S., & Murai, T. (2021). Cognitive underpinnings of multidimensional Japanese literacy and its impact on higher-level language skills. Scientific reports, 11(1), 2190. https://doi.org/10.1038/s41598-021-81909-x

Otsuka, S., & Murai, T. (2023). The unique contribution of handwriting accuracy to literacy skills in Japanese adolescents. Reading and writing, 1–26. Advance online publication. https://doi.org/10.1007/s11145-023-10433-3

SiliconFlow. (2025). *The best open source LLM for Japanese in 2025*. Retrieved from https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Japanese

Strong E., Denter C., Chan A. (2023) A new tool for learning to read Japanese on Duolingo. https://blog.duolingo.com/learning-to-read-japanese-characters/

Suzuki M. and Harada Y. (2005). Using Speech Recognition for an Automated Test of Spoken Japanese. In Proceedings of the 19th Pacific Asia Conference on Language, Information and Computation, pages 317–323, Taipei, Taiwan, R.O.C.. Institute of Linguistics, Academia Sinica.

Tsuchiya, S. (2011). Elicited Imitation and Automated Speech Recognition: Evaluating Differences among Learners of Japanese.

Wolfert, P., De Gersem, L., Janssens, R., & Belpaeme, T. (2024). Multi-modal language learning: explorations on learning Japanese vocabulary. In D. Grollman & E. Broadbent (Eds.), COMPANION OF THE 2024 ACM/IEEE INTERNATIONAL CONFERENCE ON HUMAN-ROBOT INTERACTION, HRI 2024 COMPANION (pp. 1129–1133). https://doi.org/10.1145/3610978.3640685
