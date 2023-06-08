## Text Classification & Sentiment Analysis of Consumer Reviews
Trained a machine learning model to generate text tags and give sentiment scores for game reviews to help game companies and new gamers with useful information and insights to assist companies in strategizing and help users in buying decision

## Problem Statement
### Identification of Problem
Consumer reviews are important for game developers and publishers However, there are challenges in understanding and extracting meaningful insights from them

These challenges include the sheer volume of reviews, the wide range of sentiments expressed, and the need to understand specific aspects or topics discussed

Automated techniques can be used to address these challenges and help developers improve their games

### Solution to the Problem
There are many different solutions for managing consumer reviews. Some of the most common include:

* Natural language processing (NLP) to automatically extract key information from reviews, such as sentiment analysis, aspect extraction, and categorization
* Machine learning and AI to automate the process of reviewing and filtering consumer reviews
* Sentiment analysis and opinion mining to determine the overall sentiment expressed in reviews
* Review summarization to generate concise summaries of consumer reviews
* Collaborative filtering and recommender systems to provide personalized recommendations based on consumer reviews
* User-generated content moderation to filter out spam, offensive, or misleading reviews
* User interface and visualization to develop user-friendly interfaces and visualization tools that allow users to navigate and explore consumer reviews effectively
* Feedback management systems to track, organize, and respond to consumer reviews

The best solution for a particular business or platform will depend on its specific context and requirements. However, by combining multiple approaches and continuously iterating based on feedback and data analysis, businesses can develop effective solutions for managing vast amounts of consumer reviews

## Background
### What is Steam?
Steam is a digital distribution platform for video games with over 130 million active users. It offers a large selection of games, automatic updates, cloud storage, and a social network

### Stean UI
It allows users to browse and purchase games, manage their game library, and communicate with other Steam users. The UI is designed to be easy to use and navigate, and it is highly customizable

![steam_ui](https://github.com/subhashishansda4/Game-Reviews/blob/main/misc/steam%20ui.jpg)

Here are some of the features of the Steam UI:
* A library of games: The Steam UI has a large library of games that users can browse and purchase
* A store: The Steam UI has a store where users can purchase games, DLC, and other content
* A community: The Steam UI has a built-in community where users can connect with other gamers, chat with friends, and join groups
* A workshop: The Steam UI has a workshop where users can find and download mods, maps, and other user-created content for their games
* A news section: The Steam UI has a news section where users can find news and updates about Steam and its games

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

## Workflow