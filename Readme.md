# EMNIST Character Prediction using CNN

## Overview

Developed a Convolutional Neural Network (CNN) model trained on the EMNIST ByClass dataset to recognize handwritten digits (0–9), uppercase (A–Z), and lowercase (a–z) English letters, achieving high classification accuracy. The model captures detailed spatial features using convolution and pooling layers, demonstrating strong generalization to unseen character inputs drawn on a web canvas.

## Dataset

<b>Name:</b> EMNIST ByClass<br>
<b>Size:</b> 814,255 grayscale images (697,932 training + 116,323 testing)<br>
<b>Image Size:</b> 28×28 pixels<br>
<b>Classes:</b> 62 (digits 0–9, uppercase A–Z, lowercase a–z)<br>
<b>Source:</b> NIST Special Database 19 via EMNIST

## Tech Stack

<b>Language:</b> Python<br>
<b>Frameworks:</b> PyTorch (deep learning), Flask (web backend)<br>
<b>Frontend:</b> HTML, CSS, JavaScript (Canvas API, Chart.js)<br>
<b>Libraries:</b> torch, torchvision — model training & inference<br>
&nbsp;&nbsp;&nbsp;&nbsp;PIL — image preprocessing<br>
&nbsp;&nbsp;&nbsp;&nbsp;flask — serving predictions<br>
&nbsp;&nbsp;&nbsp;&nbsp;chart.js — probability visualization in browser<br>
Environment: Runs locally in browser for live character drawing and prediction.

## Usage

Draw any digit (0–9), uppercase letter (A–Z), or lowercase letter (a–z) in white on the black canvas.

Click Predict to see the predicted character and probability chart of top predictions.

Click Clear to reset the canvas.
