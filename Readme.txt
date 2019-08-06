#use this link for embeddings:- https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

#helping functions helpers.py and cleaningtool.py are directly taken from [1]
# other helper function model.py and data.py are written by myself.
#other dependencies can be found in libraries.txt file.
# I used python[7] pytorch[2], numpy[3], matplotlib[4], NLTK[5], pandas[6]  for implementing the Architectures

#use environment.yml file to create a conda environment.

# change the label_token funtion in data.py as per the classification rask requirement

# To produce EXP 1

	-run file exp_1.ipynb
	-channge the value of out variable for the number of classes


# To produce EXP 2
	## to produce only with "statement as a feature"
	- run file exp_2_stat
	- change the vaue of bidirectional_ as per the requirement
	-channge the value of out_classes variable for the number of classes

	## to produce with statement and justification both combined as a feature
	- run file exp_2_just
	- change the vaue of bidirectional_ as per the requirement
	-channge the value of out_classes variable for the number of classes


FOR ANY ADDITIONAL HELP/INFORMATION pls. Contact:-
	--email-id : hemantya@iiitd.ac.in, raotnameh@gmail.com 
	--phoe : +91 8285207072, +91 8076523055


REFRENCES:-

1. https://github.com/mikanikos/ADA_Project
2. Adam Paszke, Sam Gross, Soumith Chintala, GregoryChanan,  Edward  Yang,  Zachary  DeVito,  Zeming  Lin,Alban Desmaison,
   Luca Antiga, and Adam Lerer,  “Au-tomatic differentiation in pytorch,” 2017
3. Travis Oliphant,  “NumPy:  A guide to NumPy,” USA:Trelgol Publishing, 2006–, [Online; accessed ¡today¿]
4.  J. D. Hunter,  “Matplotlib: A 2d graphics environment,”Computing in Science & Engineering, vol. 9, no. 3, pp.90–95, 2007
5. Bird, Steven, Edward Loper and Ewan Klein (2009).Natural Language Processing with Python.  O'Reilly Media Inc.
6. Wes McKinney,  “Data structures for statistical comput-ing in python,”  inProceedings of the 9th Python in Sci-ence Conference, St ́efan van der Walt and Jarrod Mill-man, Eds., 2010, pp. 51 – 56.
7. Python Core Team (2015). Python: A dynamic, open source programming language. Python Software Foundation. URL https://www.python.org/.
