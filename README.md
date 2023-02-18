# DataScientist_PRJ7
GitHub saving of Projet 7 of OpenClassRooms's Data Scientist formation. 

Dataset : **Home Credit Default Risk** (source : https://www.kaggle.com/competitions/home-credit-default-risk)

Notebook and Streamlit's dashbord are contained in this repository.

**Notebook** contain pipeline shaped models optimization and run, allowing prediction of client loans acceptance/refusal. 
All models include in the pipeline are classification algorithms. 

The Pipeline allow you to use it in two differents modes : **OPTIMIZATION** & **RUN**. 

* OPTIMIZATION : Optimization of hyperparameters of the selected model
* RUN : Run all models available and select the "Best" one. 

The best models is selected according to a local scoring metric named "**Cost FN**". This metric take into account the number of False Negative (FN) and False Positive (FP) producted by the model and select the one who that produces the least. Plus, in that calcul, FN have a higher coefficient (10 times more importante than FP). 

To sum : *Cost FN = 10xFN + FP*

**Streamlit's dashbord** is connected to MLFLOW best model API and deployed with Docker. 
