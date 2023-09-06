from transformers import pipeline

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

out = nlp(
    "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
    "What is the invoice number?"
)
# {'score': 0.9943977, 'answer': 'us-001', 'start': 15, 'end': 15}
print(out)

out = nlp(
    "https://miro.medium.com/max/787/1*iECQRIiOGTmEFLdWkVIH2g.jpeg",
    "What is the purchase amount?"
)
# {'score': 0.9912159, 'answer': '$1,000,000,000', 'start': 97, 'end': 97}
print(out)


out = nlp(
    "https://www.accountingcoach.com/wp-content/uploads/2013/10/income-statement-example@2x.png",
    "What are the 2020 net sales?"
)
# {'score': 0.59147286, 'answer': '$ 3,750', 'start': 19, 'end': 20}
print(out)

out = nlp("https://1.bp.blogspot.com/-BW3v9mAyzKY/XiLJePf90KI/AAAAAAABMi8/bbdAplvI-dsejDoT4ukBxj72gCYk9GZ2gCLcBGAsYHQ/s1600/license-plate-2020.jpg",
          "What are the car plate text?")

print(out)