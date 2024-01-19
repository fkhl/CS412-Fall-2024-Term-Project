# Methodology

To conduct the project, we simply tried to improve the basecode; as it already implements the simplest method to perform the project. Some parts of the basecode were left unchanged as we felt changing them would not improve the result.
The baseline code uses tf_idf methods in order to perform prompt matching. We decided to use embedded word models (word2vec was used) instead. The idea behind using embedded word models is that the prompts with most similarity will match to the question, and the prompts that have low similarity to the questions will not be matched.

We noticed that the baseline code uses decision trees as the selected model. We decided to go with a similar approach. Instead of using regular decision trees we used XGBoost, which is a complex decision tree with extreme gradient boosting. We also used an alternative method, which is random forest and linear regression with Lasso regression.

# Prompt Matching and Feature Extraction

## Instead of term frequency–inverse document frequency (tf-idf) we use word2vec for feature extraction. 

1) Data Preprocessing:
The preprocess_text function is defined to convert text to lowercase, split it into words, and remove stop words.

2) Data Preparation:
User prompts are extracted from a dataset (code2convos). It only extracts the prompts from the "user" role.

3) Word2Vec Training:
Sentences (comprising both user prompts and questions) are preprocessed, and a Word2Vec model is trained using the Word2Vec class from the Gensim library. The model is configured with a vector size of 100, a window size of 5, and a minimum word count of 1.


4) Feature Extraction for Questions:
For each question, the code obtains its Word2Vec vector by averaging the vectors of its preprocessed words. The resulting vectors are used to create a DataFrame (questions_word2vec).

5) Feature Extraction for User Prompts:
For each code, the user prompts associated with that code are processed similarly to questions. Word2Vec vectors are obtained for each prompt, and DataFrames are created for each code (code2prompts_word2vec). Empty DataFrames are printed if there are no valid prompts for a code.

6) Handling NaN Values:
NaN values in the DataFrames are replaced with zeros.

## Similarity Between Prompt and Question

Then, we calculate the cosine similarity between the Word2Vec representations of questions (questions_word2vec) and the average representations of user prompts associated with different codes (code2prompts_word2vec). The resulting similarity scores are then organized into a DataFrame (similarity_df) for further analysis.
Here is the workflow of this part:

1) Cosine Similarity Calculation:
The cosine_similarity function from scikit-learn is used to compute the cosine similarity between each code's prompts and all questions. This results in a matrix of similarity scores, where each row corresponds to a prompt for a specific code, and each column corresponds to a question.

2) Averaging Similarity Scores:
For each code (user code), we calculate the average similarity score across all prompts associated with that code. This is done by taking the mean along the rows of the similarity matrix.

3) Storing Results:
The average similarity scores for each code are stored in the code2similarity dictionary, where the code serves as the key.

4) Creating a DataFrame:
The similarity scores are organized into a DataFrame (similarity_df). Each row of the DataFrame corresponds to a question, and each column corresponds to a code. The values in the DataFrame represent the average cosine similarity score between the questions and the prompts for each code.

5) Displaying Results:
The resulting DataFrame is printed, showing the average similarity scores between questions and prompts for each code

Next, we process the similarity DataFrame (similarity_df) obtained from the previous code section. It organizes and structures the similarity scores to create a new DataFrame (question_mapping_scores) that provides a mapping between codes and their respective similarity scores for each question. 

# Feature Engineering

1) We initialize a list of keywords (keywords2search)

2) Then, next code cell processes conversations stored in code2convos and extracts various features related to user prompts and ChatGPT responses for each code. Additionally, it incorporates a pattern-based approach to identify if a user prompt contains specific error-related terms.


	2.1) Initialization
	We initialize a defaultdict of defaultdict named code2features to store features for each code. This data structure is used to store counts related to user 		prompts and ChatGPT responses.

	2.2) Iterating Through Codes and Conversations:
	Then it iterates through each code and its corresponding conversations (convs) in the code2convos dictionary.

	2.3) Counting User Prompts:
	For each user prompt in the conversations, it increments the count of user prompts (#user_prompts) for the respective code.


	2.4) Counting Keyword Occurrences:
	For each user prompt, we count the occurrences of keywords from the keywords2search list using regular expressions.

	2.5) Calculating Average Characters:
	Keep track of the total number of characters in both user prompts (prompt_avg_chars) and ChatGPT responses (response_avg_chars). It later calculates the average characters for both.

	2.6) Printing Codes with No Conversations:
	If there are no conversations (convs) for a particular code, it prints the code to the console.

	2.7) Normalization of Average Characters:
	Finally we normalize the average characters for user prompts and ChatGPT responses by dividing the total characters by the number of user prompts.

3) Then we create a Pandas DataFrame (df) from the feature information stored in the code2features dictionary. 

4) Next, we read a CSV file named "scores.csv" and store the resulting DataFrame in the variable named scores. Then some information about scores is displayed. 

5)  After that, we modify the structure of the Pandas DataFrame df that was created in a previous section. Reset_index method is used to reset the index of the DataFrame df. The  rename method is used to rename the column with the label "index" to "code". 

6)Then, we perform a left merge between two Pandas DataFrames (df and question_mapping_scores)
question_mapping_scores) is a dataframe that provides a mapping between codes and their respective similarity scores for each question, it's created earlier.

7)Next, we extend the feature merging process by incorporating information from another DataFrame named scores. The result of the merge and data cleaning operations is stored back in the temp_df. 

8)Then we set up the features (X) and the target variable (y) for a machine learning model.
	8.1) Feature Selection
		temp_df.columns[1:-1] selects all columns from the second column to the second-to-last column of the DataFrame (temp_df). These columns are considered as features for the machine learning model.
	8.2) Target Variable Selection
		temp_df["grade"] selects the "grade" column from the DataFrame (temp_df). This column is considered as the target variable for the machine learning model.
	8.3) Setting Up Features and Target Variable
		X is assigned the selected features.
		y is assigned the selected target variable.
