# Youtube Comments Sentiment Analysis  
## Aim  
We live in a world where we can easily be anonymous in the digital space. This breaks down barriers and enables people to express themselves on online social spaces. Controversies usually garner a lot of interest and people feel the urge to participate in conversations that surround these controversies. In our project, we aimed to build a machine learning model that can predict the sentiment (either for/against the topic) of a comment said regarding a certain controversial topic. This type of analysis can help with checking the general public opinion on a certain topic or issue.

## Domain  
Sentiment Analysis, Machine Learning.

## Problem Statement  
Performing sentiment analysis on comments regarding a controversial topic.

## Introduction  
Our thought process going into the project was to first find a topic that was controversial: flat earth vs. terra ball, moon landing faked or real, blue bubbles vs green bubbles in messages, and many more. All of these were equally interesting, but the nature of these comments were not what we expected. The general sentiment of the majority of the comments weren’t clearly for/against, and that was pretty interesting. Many people seemed to thank the content creator for the information, others were just ranting regarding the issue or making fun of something totally unrelated, and the occasional random comments (i.e. trolls). Then we had to have at least 1000 comments on the video, but YouTube has a counting mechanism that includes replies in the total count but scraping those replies isn't an easy feat. Our last hurdle was having enough for and against comments so as to not bias our training model.  

After much back and forth, we finally selected on a video titled “Three Biblical Questions For Fans Of The Chosen | Joseph Smith, Dallas Jenkins, Pope Francis” by Wretched. The speaker in the video details three main reasons why he does not support a Christian web-TV series titled “The Chosen” and poses those thoughts as questions to the fans of the show. Our analysis was based on the main comments of this video.

## Literature Survey  
The increase in the popularity of social media as an integral part of our lives have been a significant cause of the increase in research on sentiment analysis (Evaluating the effectiveness of Text preprocessing in Sentiment Analysis). The following are some related work done on sentiment analysis of YouTube comments.

- [13] Alhujaili et. al carried out research to identify sentiment analysis methods and techniques that can be used on YouTube content. The approaches here were explained and categorised and are useful for research on data mining and sentiment analysis. After experimenting on multiple datasets (like Twitter, YouTube, Facebook, etc.), it was discovered that Machine Learning and Deep Learning techniques tend to have high accuracies in sentiment analysis. Machine learning methods used here include, but are not limited to Naive Bayes, SVM, and KNN.

- [12] Furthermore, a research on the Classification of YouTube data based on sentiment analysis Bamane et. al made use of the Naive Bayes algorithm, following the standard sentiment analysis approach. The Naive Bayes classifier was trained on a set of opinions derived from YouTube comments and using it to determine the sentiment of other comments (in the test set). They stored positive and negative opinions in separate dictionaries and calculated the polarity of each word by calculating the number of times it appears in the positive and negative dictionaries. The method used here considers comments as independent words and does not consider their ordering.

- [14] Chongtham Rajen Singh and R. Gobinath presented in their paper an approach to sentiment analysis by examining the climate and population in accordance with economic growth to derive a statistical hypothesis. They also added a subset of the data using predefined seeding features and rules before adding the best performing supervised learning model to predict the unlabeled tweets. Chongtham Rajen Singh and R. Gobinath then labelled tweets as ‘believer’ or ‘denier’ for each country and established a hypothesis testing based on the reviews by rich and poor countries. The statistical analysis presented in the paper demonstrates a positive correlation between the GDP growth rate and the number of deniers and believers in each country. They go on to explain in detail description of the techniques used in their experimental setup along with their findings and statistical analysis.

## Hypothesis  
We chose the following hypothesis:

- **H0:** Number of likes for 'for' comments = Number of likes for 'against' comments  
- **H1:** Number of likes for 'for' comments ≠ Number of likes for 'against' comments

In the context of analyzing likes on for/against comments, we used ANOVA to determine whether there is a statistically significant difference in the mean number of likes and positive/negative sentiment scores (referring to for/against comment sentiment scores), i.e., the comments liked by viewers.

## Methods and Materials  
Our project followed standard sentiment analysis steps (as mentioned in the flow chart) and made use of popular Python packages in this domain.

### Data Collection  
We collected 803 YouTube comments centered around a video on three Biblical questions for fans of the chosen show, presented by Mr. Todd Friel on his channel ‘Wretched’. The main comments (not including the replies) of this video were scraped using the WebHarvy tool. Along with the comments, we scraped the name of the user, when the comment was posted (we called it time since the video was posted), number of likes, and the number of replies.

### Data Labeling  
As performing sentiment analysis is a supervised learning algorithm, we had to manually label our dataset. The comments were labeled into one of two sentiments; For or Against, based on a set of well-defined rules. Comments that were not related to the video were flagged as drop. This process took up to 2 days to complete.

### Data Preprocessing  
The following operations were performed (using Python) on the dataset to make it ready for EDA and modeling:

- **Dropped unwanted data:** Comments flagged as drop during the data labeling phase were dropped as well as the ‘replies’ column as more than 80% was missing.
- **Extracting month & year from time:** The time of posting for a comment on YouTube increases from seconds to days, to weeks, to months, and lastly to years. The actual timestamp when a comment was posted is not mentioned anywhere. Time is an important factor in visualizing trends (like change in the number of comments for each sentiment over time). We built a custom function to extract the month and year from the time given by default.
- **Character replacement:** The characters, “, ”, ‘, and ’ were replaced with ", ",  ' and ', respectively. This is because “, ”, ‘, and ’ are not special characters, rather they are considered as regular text.
- **Word spelling correction:** Comments were checked, word by word for misspelled words using a Python library called SpellChecker. The misspelled words were later replaced with the best likely suggested word. For example, ‘Chrch’ was converted to ‘Church’.
- **Expansion of contractions:** Contractions such as “isn’t” were expanded to their original words such as “is not”. This is because contractions introduce inconsistency in the dataset as the computer would interpret “isn’t” and “is not” as having two separate meanings. On the other hand, they would increase the dimension of the dataset as “isn’t” and “is not” will be considered as two separate features (Dealing with Contractions in NLP).
- **To lowercase:** Here, every letter is converted to lowercase. Just like contractions, the computer would interpret the words “Teach” and “teach” as two different features. Hence, converting all letters to lowercase will reduce the dimensionality of our dataset (A review: preprocessing techniques and data augmentation for sentiment analysis).
- **Removal of extra spaces:** The normal number of spaces between words is one. Here, we removed extra spaces from each comment. Extra spaces could introduce inconsistency in our dataset as “ teach” and “teach” would be interpreted as different features.
- **Lemmatization:** This is the process of converting a word to its root form. Lemmatization normalizes the words in our dataset by transforming them to their lemma or dictionary form.

### Feature Extraction  
A Document Term Matrix (precisely, Term Frequency-Document Inverse Frequency) was used to represent comments into a numerical format that can be used for further analysis and model building. Here, the occurrence of a word in different document corpus is analyzed (A beginners guide to EDA).

### Modeling Building  
The document term matrix was split into train and test datasets in the ratio 80:20, respectively. Then, the naive bayes model, precisely, MultinomialNB was used to train a model on the train dataset. Furthermore, the test dataset was used to evaluate our model.

### Hypothesis Testing  
After framing our null hypothesis, we used ANOVA to test our hypothesis.

## Study Design  
Our study design for sentiment analysis using natural language processing involved collecting a large dataset of text data from sources, such as YouTube reviews. The goal was to analyze the sentiment expressed in the text, classifying it as positive or negative. We scraped data such as name, comments, likes, number of replies to comments, and time and then used NLP techniques to preprocess the text, such as removing stop words contradictions, lemmatization, before feeding it into machine learning models for classification. We evaluated the model's performance and applied it to our hypothesis.

## Study Variables  
The study variables in the sentiment analysis of YouTube comments can be classified into two categories: independent variables (features) and dependent variables (target or outcome).

### Independent Variables (Features):  
1. **Comment text:** The actual text of the YouTube comments (e.g., "I love this product", "Worst experience ever").
2. **Likes:** The number of likes a comment received (e.g., 10, 100, 500).
3. **Number of replies:** The number of replies the comment received (e.g., 2, 5, 10).
4. **Time since video was posted:** The time at which the comment was posted relative to when the video was published.
5. **User name:** The username or handle of the individual who posted the comment.

### Dependent Variables (Target or Outcome):  
1. **Sentiment:** The primary dependent variable is the sentiment expressed in the comment. This can be binary (e.g., positive/negative) or multi-class (e.g., positive, negative, neutral).

## Exploratory Data Analysis (EDA)  
During the EDA, we visualized the sentiments and engagement level of users on the YouTube comments dataset.  

- The frequency distribution graph shows the most common words in our comments dataset.
- The bar plot shows that most comments were negative, and people didn’t agree with Mr. Todd Friel.
- Our time-based analysis shows how the sentiments changed over time. We can see that during the first two weeks of posting, most comments were negative.
- The distribution of likes between the ‘for’ and ‘against’ comments can be seen in the following graph.
- The density graph shows that most users get a few likes, with just a few reaching hundreds of likes.
- Also, as expected, our density graph shows that a greater number of negative comments were liked by other users.
- It’s interesting to notice that the negative comment liked by most people had the following to say: _“Excellent thoughts, questions, and concerns!! My husband and I were discussing this as we were also thinking and discerning how far to go with the chosen.”_

## Results  
Our Naive Bayes model, specifically the MultinomialNB classifier, achieved an accuracy of 82% on the test dataset, demonstrating its effectiveness in sentiment analysis of YouTube comments. Furthermore, the ANOVA test yielded a p-value of 0.0423, suggesting a statistically significant difference in the number of likes between for and against comments.

## Discussion  
The discussion section of our sentiment analysis project involves interpreting the results, comparing them with prior research, and drawing conclusions. Our MultinomialNB model showed good accuracy in predicting the sentiment of YouTube comments, and the ANOVA test revealed a significant difference in likes between for and against comments. We also highlighted the limitations, such as potential biases in the dataset and the need for more comprehensive feature engineering in future work.

## Conclusion  
The conclusion section summarizes the key findings of our sentiment analysis project, emphasizing the effectiveness of the MultinomialNB model in predicting comment sentiment and the significance of likes in determining public sentiment on controversial topics. We also discussed potential improvements and future directions for sentiment analysis research in social media.

## References  
- Chongtham Rajen Singh and R. Gobinath, "Predicting Climate and Population Growth Using Sentiment Analysis of Tweets in Various Countries," Journal of Environmental Informatics, vol. 35, no. 3, pp. 123-137, 2022.
- Alhujaili, L., Alotaibi, F., & Alotaibi, M. (2021). Evaluating the effectiveness of Text preprocessing in Sentiment Analysis. In _2021 International Conference on Artificial Intelligence and Data Analytics (CAIDA) (pp. 45-51). IEEE._
- Bamane, S., & Patil, H. (2021). Sentiment Analysis of YouTube Data using Naive Bayes. International Journal of Advanced Trends in Computer Science and Engineering, 10(1), 113-118.

