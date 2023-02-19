NewNLP is a simple tool useful to quickly test Latent Semantic Indexing performances.

Wanna try? Just download the code, save in your doc folder the input corpus and run the driver.py to execute queries!
NewNLP will create a file for you with a ranked list of documents sorted by relevance to the query.

Example.
Assuming you have a folder called "docs" with some text documents which you want to use to train the system and you want to execute the query 
"algorithm method" and store the result in "output.txt", than just run:

python3 unsupervised_driver.py ./docs/ "algorithm method" output.txt

And open the file "output.txt" to see the documents sorted by relevance.