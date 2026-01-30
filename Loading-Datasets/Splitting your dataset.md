
# How should you split your data

You should generally split your dataset into 3 disjoint parts:

- **Training** - the data that the model will actually learn from
- **Validation** - data that the model will be tested on as it trains (to check if the training is going as expected)
- **Testing** - data that you will evaluate your model with once training is complete



## What if my dataset is quite small

If you have only e.g. 200 data points in your dataset, then if you were to split it 70 | 15 | 15 ...

In this scenario we should generally drop the validation data and allocate that data to have a bigger test dataset

# Why do we split our dataset



# Splitting datasets (Code)

train_test_split from scikit-learn 