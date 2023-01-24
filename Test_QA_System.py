from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'what is valid english score of TOEFL?',
    'context': '''Undergraduate Admission to Zayed University. Applying to Zayed University. External Transfer. To be considered for admission, all applicants must:

complete 12 years of school

meet the high school admission requirements and EmSAT scores for the program of interest. (check the admission requirements chart)

complete the NAPO online application by selecting Zayed University as your primary choice and submit all required documents. UAE National Applicants Only.  Visit www.moe.gov.ae/AR /EServices/ServiceCard/pages/RegistrationPublic.aspx

complete the online Zayed University application. Visit application.zu.ac.ae

provide an attested General School Leaving Certificate or certificate of equivalency.

Students enrolled in an accredited higher education institution may transfer to Zayed University in accordance with the terms and conditions set by the University.

Transfer Requirements:

 Good academic standing (CGPA of 3.0)

 Valid English score of EMSAT 1250 or IELTS of 5.5 or TOEFL iBT of 71

 Meet the high school admission requirements set by Zayed University for the program of interest. (check the admission requirements chart)

 Complete the online Zayed University application. Visit application.zu.ac.ae'''
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(tokenizer)
print(res)

# # c) Get predictions
# model.get_predictions()
# # d) Get predictions with tokenizer (for example, for a question from a dataset)
# tokenizer.encode_plus('Why is model conversion important?', 
# 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.')

from transformers import pipeline

qa_model = pipeline("question-answering")
question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."
qa_model(question = question, context = context)