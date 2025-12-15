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
