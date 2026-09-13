import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments,Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix

dataset=pd.read_csv("spam.csv")
Y=[]
X=[]
for j in range(0,len(dataset["Category"])):
    if dataset["Category"].iloc[j]=="spam":
        Y.append(1)
    else:
        Y.append(0)
for k in range(0,len(dataset["Message"])):
    X.append(dataset["Message"][k])

    

X_train,X_t,Y_train,Y_t=train_test_split(X,Y,test_size=0.30,random_state=42)
X_cv,X_test,Y_cv,Y_test=train_test_split(X_t,Y_t,test_size=0.5,random_state=42)




tokenizer=AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def encodings(examples):
    return tokenizer(
        examples,
        truncation=True,
    )


train_encodings=encodings(X_train)
cv_encodings=encodings(X_cv)
test_encodings=encodings(X_test)



class SpamDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels
    def __getitem__(self,idx):
        item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset=SpamDataset(train_encodings,Y_train)
test_dataset=SpamDataset(test_encodings,Y_test)
cv_dataset=SpamDataset(cv_encodings,Y_cv)


model=AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased",num_labels=2)

def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds=np.argmax(logits,axis=-1)
    acc=accuracy_score(labels,preds)
    precision,recall,f1,_=precision_recall_fscore_support(labels,preds,average="binary")
    return {"accuracy":acc,"precision":precision,"recall":recall,"f1":f1}

data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
training_args=TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10

)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=cv_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result=trainer.train()
results=trainer.evaluate()
print("Validation Results:",results)
test_results=trainer.evaluate(test_dataset)
print("Test Results:",test_results)

predictions=trainer.predict(test_dataset)
pred_labels=np.argmax(predictions.predictions,axis=-1)
print(classification_report(Y_test,pred_labels,target_names=["ham","spam"]))
print(confusion_matrix(Y_test,pred_labels))



trainer.save_model("./spam-distilbert")
tokenizer.save_pretrained("./spam-distilbert")



