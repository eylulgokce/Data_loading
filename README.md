# Data_loading

This is a base code for downloding the dataset from the web and unziping them. You can use these datasets to process them in your classification algorithms. You can process the data either as test and train data separatly or both of them together. The available datasets are following:

- AGNews
- AmazonReviewFull
- AmazonReviewPolarity
- Dbpedia
- YelpReviewFull
- YelpReviewPolarity
- Yahoo
- IMDB
- WebKB

To run the code, create the class with defining subset, encoding and root directory (defaults are: subset="train", encoding="utf8", root="C:\Data") and call the data or target of the dateset. 

Example code for running the code:

```
  ag = AGNews("test")

  print(ag.target[:5]) #it will only print the first 5 target values of test data
```
