# Visual Question Answering for Assistive Vision

A multimodal Visual Question Answering (VQA) system designed to assist visually impaired users by answering questions about images and providing visual explanations.

---

## Overview

This project explores the use of multimodal deep learning to help visually impaired users understand visual content.

Given an image and a natural-language question, the system predicts an answer based on the visual information contained in the image.

For example:

<img width="1107" height="261" alt="Image" src="https://github.com/user-attachments/assets/c084b5f8-99f4-4847-8de5-9ac3a3e9abdf" />

The project focuses on building an end-to-end VQA pipeline using **PyTorch** and the **VizWiz** dataset.

---

## Key Features

* Multimodal Visual Question Answering using image and text inputs
* PyTorch-based deep learning model
* Attention mechanism for identifying important image regions
* Attention heatmap visualization for model interpretation
* Model training and hyperparameter tuning
* Fallback pipeline using the Gemini API
* Photo-capture suggestions when the VQA model cannot provide a reliable answer

---

## System Overview

The system consists of two main components:

```text
                ┌───────────────┐
                │     Image     │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Image Encoder │
                └───────┬───────┘
                        │
                        │
┌───────────────┐       ▼
│    Question   │ → ┌───────────────┐
└───────────────┘   │ Multimodal   │
                    │    Model     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    Answer     │
                    └───────────────┘

                            │
                   Low-confidence / failure
                            │
                            ▼
                    ┌───────────────┐
                    │ Gemini API    │
                    │ Fallback      │
                    └───────┬───────┘
                            │
                            ▼
                    Photo Suggestions
```

---

## Attention Visualization

The model uses an attention mechanism to identify image regions that contribute to its prediction.

Attention heatmaps are generated to provide a visual indication of where the model focuses when answering a question.

### Example

<!-- Add your attention heatmap image here -->

![Attention Heatmap](path/to/attention_heatmap.png)

The visualization helps provide a basic interpretation of the model's decision-making process.

---

## Dataset

The project uses the **VizWiz** dataset, which contains images and questions collected from visually impaired users.

The dataset presents several challenges compared with conventional VQA datasets, including:

* Poor image quality
* Blurry or poorly framed images
* Images containing limited useful information
* Questions that may be difficult to answer from the image alone

These characteristics make VizWiz particularly relevant to assistive-vision applications.

---

## Results

The trained model achieved:

| Metric   |     Result |
| -------- | ---------: |
| Accuracy | **61.32%** |

Model performance was evaluated after training and hyperparameter tuning.

<!-- Add training/validation curves here if available -->

![Training Results](path/to/training_results.png)

---

## Fallback Pipeline

When the VQA model fails to produce a reliable answer, the system uses a fallback pipeline based on the **Gemini API**.

Instead of returning an uncertain answer directly, the fallback system analyzes the image and provides suggestions for improving the photo capture.

For example, it may suggest:

* Moving closer to the subject
* Improving image framing
* Capturing the relevant object more clearly

This approach is intended to make the system more useful in real-world assistive scenarios where image quality can significantly affect model performance.

---

## Technologies

* **Python**
* **PyTorch**
* **Scikit-learn**
* **NumPy**
* **Pandas**
* **Gemini API**
* **Jupyter Notebook**

---

## Project Structure

```text
Visual-question-Answering-Model/
│
├── data/
├── notebooks/
├── models/
├── src/
├── results/
├── requirements.txt
└── README.md
```

> The exact structure may differ depending on the current repository organization.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Poem179/Visual-question-Answering-Model.git
cd Visual-question-Answering-Model
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

The main workflow consists of:

1. Preparing the VizWiz dataset
2. Preprocessing image and question data
3. Training the VQA model
4. Evaluating model performance
5. Generating attention visualizations
6. Using the fallback pipeline for unsuccessful predictions

Refer to the notebooks and source code for implementation details.

---

## Limitations

The project has several limitations:

* Performance is affected by the quality of input images.
* VQA accuracy remains limited on difficult or ambiguous questions.
* Attention heatmaps provide an indication of model focus but should not be interpreted as a complete explanation of model reasoning.
* The Gemini-based fallback pipeline depends on an external API.

---

## Future Improvements

Potential improvements include:

* Improving model architecture and multimodal fusion
* Exploring stronger pretrained vision-language models
* Improving handling of low-quality images
* Developing a more robust confidence estimation mechanism
* Building a complete real-time assistive application

---

## Author

**Nguyen Hoang Duy**

Bachelor of Engineering in Information Technology
Saigon University

Machine Learning · Computer Vision · Multimodal Learning
