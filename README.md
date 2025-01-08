# Bi-Modal Empathetic Response Generation
This project is implementation of Bi-Modal Response Generator model using audio and text modalities. the architecture of this model is showen in below picture. this model is combined of two models with diffrent tasks that are fine-tunined with their own dataset:
- task-1: Empathetic Response Generation using EmpatheticDialogues dataset
- task-2: Bi-Modal Emotion Recognition using MELD dataset
<p></p>

After combining two models to get Bi-Modal Empathetic Response Generation, model is trained using [BiMEmpDialogues](https://github.com/zolfaShefreie/Multi-model_Empathetic_Dialogue_Dataset) dataset.

![artitecture - Copy-multi model response generation](https://github.com/user-attachments/assets/4a4bcc2a-57d0-42ec-ba2b-08bfc31a4976)

## SetUp
### Prerequisites for running
First of all run below code to clone the repository
```
git clone https://github.com/zolfaShefreie/empathetic_response_generation.git
```
Make an envierment and run below command to install all required packages
```
pip install -r requirements.txt
```
<b>Make sure you complete ".env" file and prepare empathy model checkpoints</b>
### Train Models
### Evaluate Models
### Comunicate With Model
