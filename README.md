# Enron-data-poi-identifier

In 2000, Enron was one of the largest companies in the US, but soon it went bankruptcy due to corporate fraud. The interested reader can click on the following link to view details: https://en.wikipedia.org/wiki/Enron

The goal of this project is to perform data cleaning and transformation methods, along with various machine learning algorithms, to build identifiers to predict who is a POI (person of interest) based on the Enron data set. POI means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. 

* To view the project report, the reader could open either html file, or Final_proj.ipynb. Remember to include ml_map.png file to view reports.

* feature_format.py and tester.py are only utilities. If the read would like to run Final_proj.ipynb, please include these files.

* final_project_dataset.pkl is the data set used in this project.

* The Enron data set used to build the machine learning prediction model in this report contains 146 personal data, with 21 features. These features are:
A. POI (Person of interst) label:
'poi': '1' means this person is POI. '0' means this person is NOT a POI. In this training data set there're 18 POIs.
B. Financial features (all units are in US dollars):
B.1 Payments
Features include: 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments'.
B.2 Stock Value
Features include: 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value'.
C. Email features:
Features include: 'email_address', 'to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'.
'email_address' is a text string; units other than 'email_address' are number of emails messages.
