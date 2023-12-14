# news-headline-correction

## Abstract:
This project addresses the issue of inaccurate headlines in news media by developing a model to classify headline-article alignment and suggest improvements
for misaligned cases. The focus is on combating misinformation, particularly in
topics like politics, crime, and clickbait articles. The study employs a nuanced
approach to model construction and scoring, emphasizing the F1 score to handle
class imbalance. A generative model, specifically fine-tuning a pretrained model for
headline generation, is introduced, marking a notable advancement. The headline
classification model demonstrates robust performance, with BERT fine-tuned on
non-summarized text consistently outperforming in various scenarios. Evaluation
of the headline generation models favors T5 for its well-rounded performance in
relevance, readability, and style.

## Details:

### Data:
Sourced from the Fake News Challenge (FNC) - http://www.fakenewschallenge.org/

fnc-1-master - original FNC github repo
