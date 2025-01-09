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
Download [finetuned dpr](https://huggingface.co/Shefreie/best_dpr_finetuned) and [empathy's classifiers](https://github.com/declare-lab/exemplary-empathy) and path of them to .env file.

<b>Make sure you complete ".env" file and prepare empathy model checkpoints</b>
### Train Models
You can use run "prepare_dataset.py" script to preprocess MELD, EmpatheticDialogues, and BiMEmpDialogues datasets. however train script run the preprocessing for specefic dataset. 
```
python prepare_dataset.py --dataset_name EmpatheticDialogues
```
To skip preprocessing stage you can download the file from this [link](https://huggingface.co/datasets/Shefreie/BiModalResponseGeneration_project_processed_dataset) and put them on "./data/cache" directory.
```
python train_model.py --model EmotionalTextualResponseGenerator --number_of_epochs 20 --early_stopping_patience 2 --logging_steps 500 --per_device_train_batch_size 4 --per_device_eval_batch_size 4
```
To skip training you can put huggingFace links of trained on .env file.

<table>
  <tr>
    <td><b>Model</b></td>
    <td><b>HuggingFace Link</b></td>
  </tr>
  <tr>
    <td>BiModalResponseGenerator</td>
    <td>Shefreie/BiModal_empathetic_response_generation</td>
  </tr>
  <tr>
    <td>BiModalEmotionClassifier</td>
    <td>Shefreie/BiModal_emotion_recognition</td>
  </tr>
  <tr>
    <td>EmotionalTextualResponseGenerator</td>
    <td>Shefreie/Multi_learning_emp_response_generation_emotion_classification</td>
  </tr>
  <tr>
    <td>TextualResponseGenerator</td>
    <td>Shefreie/Knowledge_based_examplar_empathetic_response_generation</td>
  </tr>
</table>

### Evaluate Models
For evaluating with test dataset you can run the "evaluate_model.py" script. 
```
python evaluate_model.py --model EmotionalTextualResponseGenerator --logging_steps 500  --per_device_eval_batch_size 16 --generation_config_path "./generation_config_greedy.json"
```
For evaluate your generated responses with [FED](https://github.com/Shikib/fed), [DynaEval](https://github.com/e0397123/DynaEval) and EmpathyPresent, you can run the blew commant. before running make sure you downloaded DynaEval models and put its path to .env file.
```
python evaluate_more.py --model own --result_path {result_path} --dataset_name EmpatheticDialogues --batch_size 16
```

### Comunicate With Model
```
python generate_res.py --model BiModalResponseGenerator --generation_config_path generation_config.json --conversation_path conv.json
```
