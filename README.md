# AI Tools for Document Analysis and Discovery

**Conference**: Billion Lives Symposium, UCSF, April 2025

**Presenter:** Geoffrey Boushey  

**Department:** Data Science and Open Scholarship, UCSF Library

**Title**: AI Tools for Document Analysis and Discovery

This repository contains materials for the presentation on the use of AI tools in document analysis and discovery. The content showcases how AI and ML tools can be applied to analyze various media files, such as videos and images, particularly in the context of tobacco marketing and health outreach campaigns.

**Conference Website and Recording**

- Billion Lives Symposium: https://tobacco.ucsf.edu/it’s-about-billion-lives-annual-symposium
- Recording (This presentation starts at 4:10:00): https://ucsf.box.com/s/8qfymrfofdvnalsf4qj2u3zwyvbgp7we

## Overview

In this presentation, we demonstrate the application of AI tools to analyze and extract insights from **three videos** and **one image** (links below). These media files include:

- A vaping company’s cartoon ad
- A youth anti-smoking campaign produced by a tobacco company
- An anti-smoking ad produced by the CDC
- A handwritten letter to a tobacco company

The goal is to explore how AI can be used to transcribe audio, detect objects in images, classify documents, and analyze sentiment in these materials. Additionally, we’ll focus on open-source tools that facilitate these tasks.

## Video & Image Links

- [VoltMan - Episode 1; Smokeless Image Electronic Cigarettes (Video)](https://archive.org/details/tobacco_yqwg0225)
- [Think. Don’t Smoke - Philip Morris Ad (Video)](https://archive.org/details/tobacco_wbr62a00)
- [CDC: Tips from Former Smokers - Fred W. (Video)](https://www.youtube.com/watch?v=CuPk1cLrq_s)
- [In My Own Handwriting (Image)](https://www.industrydocuments.ucsf.edu/tobacco/docs/#id=ytxk0091)

## Technologies

1. **Speech-to-Text Transcription**
   - Tools for converting spoken language in videos to text, such as **Whisper** (OpenAI) and **YouTube API** for automatic transcription.
     - [Whisper - OpenAI](https://openai.com/research/whisper)
     - [YouTube API Documentation](https://developers.google.com/youtube/v3)

2. **Optical Character Recognition (OCR)**
   - Using tools like **Tesseract** and **Google Vision API** for recognizing and extracting text from images and video frames.
     - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
     - [Google Cloud Vision API](https://cloud.google.com/vision)
     - [Amazon Rekognition](https://aws.amazon.com/rekognition/)

3. **Image Labeling & Object Detection**
   - Introduction to image analysis with **TensorFlow Object Detection API** and **Google Cloud Vision API** for identifying and labeling objects within visual content.
     - [TensorFlow Object Detection API](https://tensorflow.org/lite/models/object_detection/overview)
     - [Google Cloud Vision API](https://cloud.google.com/vision)

4. **Document Classification & Sentiment Analysis**
   - Classification techniques using **Zero-Shot Classifier** from HuggingFace and sentiment analysis with **VADER** for analyzing the tone and intent of documents.
     - [HuggingFace - Zero-Shot Classification](https://huggingface.co/transformers/main_classes/pipelines.html#zero-shot-classification)
     - [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)

5. **Generative AI for Text Analysis**
   - Exploring **OpenAI's GPT models** for more complex document classification and sentiment analysis tasks.
     - [OpenAI GPT Models](https://openai.com/gpt-3)
   - Exploring **VERSA API and Chat** for Image and Text Analysis.
     - [OpenAI GPT Models](https://ai.ucsf.edu/platforms-tools-and-resources/ucsf-versa)

6. **Custom Machine Learning Models**
   - Building task-specific models for document analysis and their benefits and challenges.
     - [Scikit-Learn - Machine Learning Library](https://scikit-learn.org/stable/)

## Code Samples

- [GitHub Repository for Presentation Code and Artifacts](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research)  
  All the code and resources used in the presentation are available here, including notebooks for transcription, OCR, object detection, and sentiment analysis.

- [Whisper AI Transcription Notebook](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research/blob/main/Whisper_AI_Transcript.ipynb)  
  Open-source transcription tool for audio and video files.

- [YouTube Transcription Notebook](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research/blob/main/Youtube_Transcript.ipynb)  
  Extracts transcriptions from YouTube videos using the YouTube API.

- [Tesseract OCR Notebook](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research/blob/main/Python_Tesseract_OCR.ipynb)  
  OCR tool for extracting text from images.

- [TensorFlow Object Detection Notebook](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research/blob/main/TensorFlow_Labels.ipynb)  
  Object detection model to label images using TensorFlow.

- [HuggingFace Zero-Shot Classifier Notebook](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research/blob/main/HuggingFace_ZeroShot_Classifier.ipynb)  
  Classification of documents into predefined categories.

- [VADER Sentiment Analysis Notebook](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research/blob/main/Vader_Sentiment.ipynb)  
  Sentiment analysis using the VADER tool for social media and text data.

- [VERSA OCR, Sentiment, Classification, Summarization](https://github.com/geoffswc/Libguide-AI-Tools-Archival-Research/blob/main/versa-text-image.ipynb)    
  Image Embedded Text Extraction, Document Classification, Sentiment Analysis, and Document Summarization

## Additional Resources

- **UCSF Library Data Science and Open Scholarship**  
  - [Consulting, Workshops, Newsletter, Events](https://library.ucsf.edu/data-science)
  - [Data Science and Open Scholarship Guides](https://guides.ucsf.edu/data-science)

---

