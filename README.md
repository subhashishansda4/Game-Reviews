## Text Classification & Sentiment Analysis of Consumer Reviews
Trained a machine learning model to generate text tags and give sentiment scores for game reviews to help game companies and new gamers with useful information and insights to assist companies in strategizing and help users in buying decision

## Problem Statement
### Identification of Problem
Consumer reviews are important for game developers and publishers However, there are challenges in understanding and extracting meaningful insights from them

These challenges include the sheer volume of reviews, the wide range of sentiments expressed, and the need to understand specific aspects or topics discussed

Automated techniques can be used to address these challenges and help developers improve their games

### Solution to the Problem
There are many different solutions for managing consumer reviews. Some of the most common include:

* **Natural language processing (NLP)** to automatically extract key information from reviews, such as sentiment analysis, aspect extraction, and categorization
* **Machine learning and AI** to automate the process of reviewing and filtering consumer reviews
* **Sentiment analysis and opinion mining** to determine the overall sentiment expressed in reviews
* **Review summarization** to generate concise summaries of consumer reviews
* **Collaborative filtering and recommender systems** to provide personalized recommendations based on consumer reviews
* **User-generated content moderation** to filter out spam, offensive, or misleading reviews
* **User interface and visualization** to develop user-friendly interfaces and visualization tools that allow users to navigate and explore consumer reviews effectively
* **Feedback management systems** to track, organize, and respond to consumer reviews

The best solution for a particular business or platform will depend on its specific context and requirements. However, by combining multiple approaches and continuously iterating based on feedback and data analysis, businesses can develop effective solutions for managing vast amounts of consumer reviews

## Background
### What is Steam?
Steam is a digital distribution platform for video games with over 130 million active users. It offers a large selection of games, automatic updates, cloud storage, and a social network

### Stean UI
It allows users to browse and purchase games, manage their game library, and communicate with other Steam users. The UI is designed to be easy to use and navigate, and it is highly customizable

![steam_ui](https://github.com/subhashishansda4/Game-Reviews/blob/main/misc/steam%20ui.jpg)

Here are some of the features of the Steam UI:
* **A library of games**: The Steam UI has a large library of games that users can browse and purchase
* **A store**: The Steam UI has a store where users can purchase games, DLC, and other content
* **A community**: The Steam UI has a built-in community where users can connect with other gamers, chat with friends, and join groups
* **A workshop**: The Steam UI has a workshop where users can find and download mods, maps, and other user-created content for their games
* **A news section**: The Steam UI has a news section where users can find news and updates about Steam and its games

### Steam Game ID
The Steam game ID is a unique identifier for a game on Steam. It is used to identify the game in Steam APIs, such as the User Reviews API. To get user reviews for a game, you need to know the game's Steam ID

You can find the Steam ID for a game by looking at the URL of its store page. The Steam ID will be at the end of the URL, after the "appid=" part

`httos://store.steampowered.com/appreviews/<appid>?json=1`

![steam_game_id](https://github.com/subhashishansda4/Game-Reviews/blob/main/misc/steam%20game%20id.jpg)

### Steam API
The Steam API is a set of functions that allow developers to access data from Steam, such as game information, user profiles, and chat logs

The API is divided into two parts: the Web API and the Steamworks API. The Web API is designed for developers who want to create web applications that interact with Steam\
The Steamworks API is designed for developers who want to create games that interact with Steam

![steam_api](https://github.com/subhashishansda4/Game-Reviews/blob/main/misc/steam%20api.jpg)

![steam_api_elements](https://github.com/subhashishansda4/Game-Reviews/blob/main/misc/steam%20api%20elements.jpg)

## Raw Data
With the help of the Steam API took the data of recent consumer reviews of five different games. All of the computing is done on a sample size of 2000 however, the population was much larger (~2,00,000)

## Initial EDa
count of positive reviews\
![positive](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/positive.jpg)

count of positive reviews per game\
![positive_per_game](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/positive_per_game.jpg)

scatter plot (votes v/s score)\
![votes_score](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/votes_vs_score.jpg)

month-wise review count\
![month](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/month.jpg)

year-wise review count\
![year](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/year.jpg)

day-wise review count\
![day](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/day.jpg)

## Data Preprocessing
Once the dataset is collected, it needs to be pre-processed to ensure its quality and suitability for analysis\
\
This involves several steps, including:
* Text Cleaning: Removing any irrelevant characters, symbols, or special characters from the reviews. Also replacing numbers and symbols with its equivalent text counterpart
* Stopwords Removal: Removing common words that do not carry significant meaning, such as "the," "is," "and," etc. Words like generic names, full forms of abbreviations and also contractions
* Tokenization: Breaking down the reviews into individual words or tokens to prepare them for further analysis
* Lemmatization or Stemming: Reducing words to their base or root form to handle variations of the same word. For example, converting "running," "runs," and "ran" to the base form "run."

## Feature Engineering
### Positive & Negative Words
Created a list of positive and negative words\
[positive_words](https://gist.github.com/mkulakowski2/4289437)\
[negative_words](https://gist.github.com/mkulakowski2/4289441)

### Scoring Functions
* **Polarity**\
  (positive_score - negative_score) / (positive_score + negative_score)\
  [-1, 1]

* **Subjectivity**\
  (positive_score + negative_score) / (len(words))\
  [0, 1]

* **Average Word Length**\
  characters_count / words_count

* **Average Word per Sentence**\
  words_count / sentences_count

* **Fog Index**\
  0.4 * (average_sentence_length + complex_word_percent)

* **Personal Pronouns**
* **Syllable Count**
* **Complex Word Count**
* **Complex Word Percent**

### Sentiment Analysis
Performed sentiment using a pre=trained model named "distilbert-base-uncased-finetuned-sst-2-english"

### Hashtag Generation
Used Word2Vec embeddings for game related words like (gameplay, controls, sound, graphics)\
Calculates the similarity between each word and suggests the words most related to one of the game related words

## Processed EDA
word cloud 1\
![word_cloud_1](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/word_cloud_1.jpg)

word cloud 2\
![word_cloud_2](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/word_cloud_2.jpg)

polarity score per game\
![polarity_score](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/polarity_score.jpg)

positive-wise subjectivity score\
![subjectivity_score](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/subjectivity_score.jpg)

## Vectorization
First I created a TfidfVectorizer object. A Tfidf Vectorizer is a statistical model that transforms a collection of text documents into a matrix of TF-IDF values

Then I create word embeddings for every word using Word2Vec and feed them into the Tfidf Vectorizer object to generate a matrix of arrays

## Model Evaluation
I prepared the dataset by dividing it into training and testing datasets

I then imported necessary libraries and models for model selection and evaluation

I initialized the classifiers and created a list called models containing them

I performed a grid search for each model to find the best hyperparameters

I then fitted the grid search on the training data and made predictions on the testing data

Classification reports, confusion matrices, and other evaluation metrics were printed for both tags and sentiment tasks

![decision_tree](https://github.com/subhashishansda4/Game-Reviews/blob/main/console/decision_tree.jpg)

![linear_discriminant](https://github.com/subhashishansda4/Game-Reviews/blob/main/console/linear_discriminant.jpg)

![naive_bayes](https://github.com/subhashishansda4/Game-Reviews/blob/main/console/naive_bayes.jpg)

![random_forest](https://github.com/subhashishansda4/Game-Reviews/blob/main/console/random_forest.jpg)

![support_vector](https://github.com/subhashishansda4/Game-Reviews/blob/main/console/support_vector.jpg)

## Final Output
![tags](https://github.com/subhashishansda4/Game-Reviews/blob/main/console/tags_pred.jpg)

![sentiment](https://github.com/subhashishansda4/Game-Reviews/blob/main/console/sentiment_pred.jpg)

## Final EDA
![tags_count](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/tags_count.jpg)

![pred_tags_count](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/pred_tags_count.jpg)

![sent_count](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/sent_count.jpg)

![pred_sent_count](https://github.com/subhashishansda4/Game-Reviews/blob/main/graphs/pred_sent_count.jpg)

## Conclusion
### Significance & Limitations
1. Customer Insights\
  Analyzing consumer reviews can provide valuable insights into the opinions, preferences, and experiences of Destiny 2 players\
  By classifying the reviews and extracting sentiment, the project can help identify common themes, positive and negative aspects of the game, and specific areas of improvement

2. Product Improvement\
   By categorizing consumer reviews into different topics (tags) and analyzing sentiment, the project can help identify specific areas of the game that players appreciate or find problematic

3. Marketing and Reputation Management
   Analyzing consumer sentiment can provide insights into the overall reputation of Destiny 2 among players\
   Positive sentiment can be leveraged for marketing purposes, highlighting the game's strengths and positive aspects in promotional materials and advertisements

4. User Feedback Analysis
   Steam reviews provide a platform for players to express their opinions and provide feedback on the game. Analyzing and categorizing this feedback can help identify recurring issues, bugs, or glitches that players encounter, allowing developers to respond and address these concerns effectively

5. Community Engagement
   Understanding the sentiments and preferences of players can facilitate better engagement with the gaming community. By analyzing and categorizing player feedback, developers can identify enthusiastic supporters, influencers, and community advocates


6. Competitive Analysis
   Analyzing the sentiment and reviews of Destiny 2 in comparison to other similar games can provide insights into the game's competitive position


### Limitations & Future Work
The project code performs text processing, feature engineering, and analysis on the Destiny 2 consumer reviews from Steam. However, there are several limitations and areas for improvement in the code:

1. Hard-coded file paths: The code includes hard-coded file paths for input and output files, such as 'output/raw_df.csv' and 'graphs/positive.jpg'. It would be better to make these paths configurable or parameterize them to improve flexibility and ease of use

2. Lack of data preprocessing techniques: The code performs basic text processing steps, such as lowercase conversion, special character handling, and stop word removal. However, it could benefit from additional preprocessing techniques like spell checking, handling contractions, and removing HTML tags (if present)

3. Lack of error handling: The code does not include sufficient error handling mechanisms. It would be helpful to include appropriate error handling, such as exception handling, to handle potential errors during file operations, model loading, or other critical parts of the code

4. Limited use of machine learning models: The code primarily focuses on descriptive analysis and feature engineering, but it does not incorporate machine learning models for text classification or sentiment analysis. Integrating machine learning models like Naive Bayes, SVM, or deep learning models could enhance the accuracy and predictive power of the analysis

5. Limited evaluation and validation: The code does not include an evaluation or validation process for the generated features or sentiment analysis. It would be beneficial to incorporate techniques such as cross-validation or holdout validation to assess the model's performance and generalize its findings

6. Scalability and efficiency: Depending on the size of the dataset, some operations in the code, such as tokenization and feature generation, may become slow or memory-intensive. It would be necessary to consider efficient algorithms or libraries (e.g., spaCy) to handle larger datasets more effectively
