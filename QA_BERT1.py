from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline
import os
import json
# from transformers import BertConfig
# from transformers import BertForSequenceClassification
# from transformers import BertForTokenClassification
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad') 

contexts = []

contexts.append("""IT Support can be contacted through following channels
Register your IT requests or issues by logging to the portal 
              https://sanad.zu.ac.ae
Send an email to IT.ServiceDesk@zu.ac.e
Contact through phone at 02 599 3666 for Abu Dhabi and 04 402 1777 for Dubai
""")

contexts.append("""Happiness Center is open for support from 8 am – 6 pm.""")
contexts.append("""To request for lab support, please get in touch with the respective lab support personnel as mentioned below:
Dubai Campus
CACE Lab – Marcus Tolledo
CCMS Lab – Fathima AlHammadi
CTI – Emerson Bautista
Abu Dhabi Campus 
CACE Lab – Ayesh Ghanim
CCMS Lab – Amar Moraje
CTI – Fatima AlKarbi
""")
contexts.append("""Support for Master classes is provided on request basis, which can be registered in SANAD using the portal https://sanad.zu.ac.ae.
The request has to be registered at least one week in advance.
""")
contexts.append("""Yes, we can assist you through remote support. You need to just open a request in SANAD through https://sanad.zu.ac.ae or send an email to IT.ServiceDesk@zu.ac.ae

""")
contexts.append("""Please raise a service request in SANAD as detailed in point 5 aboveSANAD and Choose “ I need a new service” to request for transferring extension to mobile phone.""")
contexts.append("""ZU has licenses for Webex, ZOOM and Microsoft Teams collaboration software.
ZOOM is primarily used for academic purposes, Webex for administrative and Microsoft Teams as a backup.
A request can be raised to provide access to any of the collaboration tools to arrange and conduct meetings. The only requirement for these tools to be provisioned is you should have a ZU ID number. 
""")
questions=[]
questions.append("How can I contact IT Support?")
questions.append("Timings of happiness center for IT support")
questions.append("How do I request lab support? ")
questions.append("Support for evening classes i.e Master or after working hours ")
questions.append("Can you help me fix technical issues at home?")
questions.append("How do I divert my extension to my mobile phone?")
questions.append("How do I arrange & conduct meetings when I am working remotely?")
contexts_str = ''
for context in contexts:
    contexts_str+= context
# print(contexts)
# print(questions)
# tokenizer.encode(questions[0], truncation=True, padding=True)
# nlp = pipeline('question-answering',model,tokenizer=tokenizer)
# # print(tokenized_q1)
# result = nlp({
#     'question': questions[1],
#     'context': contexts_str           
# })

# print(result)

train_json = {"data":[{"title":"IT support & general questions", "paragraphs":[],}]}
# print(train_json["data"][0]['paragraphs'])
for i in range(len(contexts)):
    data_dict_paragraphs={
        "qas":[{"question": questions[i], "id": i, "answers":[{"text": contexts[i], "answer_start": 0}], "is_impossible": False}], 
        "context": contexts[i],
        }
    train_json["data"][0]['paragraphs'].append(data_dict_paragraphs) 
# print(train_json)

answers =  [{"text": contexts[i], "answer_start": 0, "answer_end":len(contexts[i])-1} for i in range(len(contexts))]
print(answers[3])
# train_encodings = tokenizer(contexts, questions, truncation=True, padding=True)
# print(train_encodings.char_to_token(0, answer_start)
start_positions = []
end_positions = []
# for context in contexts:
#     start_positions.append(train_encodings(0,0))
#     end_positions.append(train_encodings(0,len(context)))

# train_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
# print(train_encodings)