# Natural Language Processing notebooks
Repository destinated for work only for university NLP studies taken in 2021-I in the institution Fundación Universitaria Konrad Lorenz for the specialization "Especialización en analítica estratégica de datos" with [Viviana Márquez](https://github.com/vivianamarquez), an amazing teacher with great sense of humor.
 
Every notebook here is made in jupyter notebooks, a tool that comes with [the anaconda paltform](https://www.anaconda.com/products/individual) and [the python language](https://www.python.org/about/). Each notebook will have a short description, the topic covered, and the link to the jupyter notebook in the [nbviever tool](https://nbviewer.jupyter.org/), a tool that makes easy the visualization of large notebooks, or visualizing plot outputs within the notebook. You can also the .ipynb notebooks in the "Talleres" folder. (Some of the notebooks are documented in "Spanglish")

## [Assignment 2](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%202%20-%20Fabi%C3%A1n%20Castro.ipynb) - Text Acquisition
This notebook contains the instructions for:
* Uncompressing
* Reading
* Processing
pdf files, takes a zip file containing different pdf files, reads and outputs the number of words in each of the pdf files. Then in the last section, prints the name of the pdf file with the greatest number of words.

## [Assignment 3](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%203%20-%20Fabi%C3%A1n%20Castro.ipynb) - Web Scraping
This notebook extracts the information for 10 animals taken from Wikipedia through web scraping:
* Read the first paragraph
* Prints the words in **bold** and *italic*
* Replaces the special characters with an asterisk (*)

## [Assignment 4](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%204%20-%20Fabi%C3%A1n%20Castro.ipynb) - Pre-processing and Feature Engineering
This notebook takes a dataset containing the dialogs of the eighth episode of the sixteenth season of south park "Sarcastaball":
* Reads the csv file containing the dialogs
* Makes pre-processing of the text
* Creates a vectorial representation for each dialog with `CountVectorizer` from [scikit learn library](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for conversion of text documents into a matrix of token counts.

## [Assignment 5](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%205%20-%20Fabi%C3%A1n%20Castro.ipynb) - TF-IDF
This notebook takes a dataset describing bob sponge characters:
* Makes pre-processing of the character's description
* Creates a matrix of features based on the TF-IDF algorithm
* Prints a heatmap based on the cosine similarity of each of the characters
* Shows a small analysis about the most similar characters and the most different

## [Assignment 7](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%207%20-%20Fabi%C3%A1n%20Castro.ipynb) - Text classification
This notebook displays a small classification exercise:
* Dataset about videogame/jewel reviews
* Training/testing dataset splitting
* Makes pre-processing of the reviews in the training dataset
* Using `MLPClassifier` model for classification ([from scikit learn library](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html))
* Assessing the model's precision with some metrics of [scikit learn metrics module](https://scikit-learn.org/stable/modules/model_evaluation.html) 
* Displaying the confu  sion matrix
* Printing the most representative words for each category (videogames/Jewelry)

## [Assignment 8](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%208%20-%20Fabi%C3%A1n%20Castro.ipynb) - Text clustering
This notebook uses the same dataset used for classification (videogames/jewelry dataset):
* Makes pre-processing of the reviews
* Creates the vectorized representation with TF-IDF
* Creating 20 distinct [Kmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) models and choosing the best K through the [elbow method](https://www.google.com/search?q=the+elbow+method+kmeans&rlz=1C1CHZN_enCO941CO941&oq=the+elbow+method+kmeans&aqs=chrome..69i57.5001j0j7&sourceid=chrome&ie=UTF-8).
* Training the model with the number of clusters chosen
* Prints the number of observations of each cluster
* Shows a small description of each cluster, and the topics (of text) covered in each
* Visualization of the data using [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* Comparison of results with the model of Assignment 7

## [Assignment 9](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%209%20-%20Fabi%C3%A1n%20Castro.ipynb) - Topic Modeling with [LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
This notebook uses the unsupervised model [LDA (Latent Dirichlet allocation)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) with the same dataset of the last two assignments (videogames/jewelry):
* Makes pre-processing of the reviews
* Creates LDA model
* Plotting of the LDA model results
* Prints the most representative documents for each topic

## [Assignment 10](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%2010%20-%20Fabi%C3%A1n%20Castro.ipynb) - Visualization of NLP (Visualization of a Youtube channel's videos comments)
This notebook uses the [google API for python](https://github.com/googleapis/google-api-python-client/blob/master/docs/start.md):
* Extracts comments for videos of a single channel
* Makes pre-processing of the comments
* Creates and displays [word clouds](https://pypi.org/project/wordcloud) with different background images

## [Assignment 11](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%2011%20-%20Fabi%C3%A1n%20Castro.ipynb) - Twitter API data
This notebook uses the twitter API from the [tweepy](https://www.tweepy.org/) python's module
* Gets data from twitter using the `api.search_full_archive` [method](https://developer.twitter.com/en/docs/twitter-api/premium/search-api/quick-start/premium-full-archive)
* Cleans the text
  * Extracts links
  * hashtags
  * mentions
* Makes pre-processing of the cleaned text
* Creates a vectorized representation using TF-IDF
* Creates a heatmap of tweets (shows the relationship of tweets)

## [Assignment 12](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Talleres/Taller%2012%20-%20Fabi%C3%A1n%20Castro.ipynb)
 - Sentiment Analysis
This notebook uses the `SentimentIntensityAnalyzer` method from [vaderSentiment module](https://pypi.org/project/vaderSentiment):
* Extracts information from twitter (this time with the [snscrape library](https://github.com/JustAnotherArchivist/snscrape))
  * Gets 2000 tweets about [overwatch twitter](https://twitter.com/playoverwatch)
* Cleans text (removes links, mentions, hashtags)
* Assigns a value of sentiment (positive, negative) to each comment
* Prints the most negative comment (expect extreme toxicity) and the most positive (expect a lot of love and mercy mains)

## [Final Proyect](https://nbviewer.jupyter.org/github/KstroO/NLPUni/blob/main/Proyecto/Modelado%20de%20temas%20-%20Noticieros.ipynb)
With the social network's scrapper [snscrape](https://github.com/JustAnotherArchivist/snscrape) gets data from different news channels in Colombia for the years 2019 and 2020. Later makes a comparison of tweets of the two years using [LDA](https://radimrehurek.com/gensim/models/ldamodel.html) and other graphics like emojis and wordcloud.
