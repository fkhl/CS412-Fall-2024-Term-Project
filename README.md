## Methodology

We have two files in this repository to represent two different approaches we took. Both of the files are our project submission.

For this project, we tried to improve the base code given to us. In the base code, tf_idf was used for prompt matching. In our implementation, we've used embedded word models (word2vec in our case) for that. We used embedded word model as we thought it was more suitable as the model would match the prompt to the most similar question.

We also used two different models, XGBoost and random forest and linear regression with Lasso regression. The base code uses decision trees, so we decided to use models with a similar idea.


# For CS412 Term Project (first version)
## 1 - Prompt matching with questions
Instead of term frequency–inverse document frequency (tf-idf) we use word2vec for feature extraction. 
#### 1.1.
```
def preprocess_text(text):
    # Convert text to lowercase and split it into words
    words = text.lower().split()
    # Remove stop words
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return words
keywords = []
prompts = []
...
```
Breakdown of this code cell:
###### 1.1.1. Data Preprocessing:
The preprocess_text function is defined to convert text to lowercase, split it into words, and remove stop words.

###### 1.1.2. Data Preparation:
User prompts are extracted from a dataset (code2convos). It only extracts the prompts from the "user" role.

###### 1.1.3. Word2Vec Training:
Sentences (comprising both user prompts and questions) are preprocessed, and a Word2Vec model is trained using the Word2Vec class from the Gensim library. The model is configured with a vector size of 100, a window size of 5, and a minimum word count of 1.

###### 1.1.4. Feature Extraction for Questions:
For each question, the code obtains its Word2Vec vector by averaging the vectors of its preprocessed words. The resulting vectors are used to create a DataFrame (questions_word2vec).

###### 1.1.5. Feature Extraction for User Prompts:
For each code, the user prompts associated with that code are processed similarly to questions. Word2Vec vectors are obtained for each prompt, and DataFrames are created for each code (code2prompts_word2vec). Empty DataFrames are printed if there are no valid prompts for a code.

###### 1.1.6. Handling NaN Values:
NaN values in the DataFrames are replaced with zeros.


#### 1.2.

Then, we calculate the cosine similarity between the Word2Vec representations of questions (questions_word2vec) and the average representations of user prompts associated with different codes (code2prompts_word2vec). The resulting similarity scores are then organized into a DataFrame (similarity_df) for further analysis.

```
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between questions_word2vec and each code's prompts_word2vec
code2similarity = {}

for code, prompts_df in code2prompts_word2vec.items():
    similarity_scores = cosine_similarity(prompts_df, questions_word2vec)
    # Average similarity scores across prompts for each code
    avg_similarity_score = similarity_scores.mean(axis=0)
    code2similarity[code] = avg_similarity_score

# Create a DataFrame to store the similarity scores
similarity_df = pd.DataFrame(code2similarity, index=questions)

# Display the resulting DataFrame
print(similarity_df)
```
Here is the workflow of this part:
###### 1.2.1. Cosine Similarity Calculation:
The cosine_similarity function from scikit-learn is used to compute the cosine similarity between each code's prompts and all questions. This results in a matrix of similarity scores, where each row corresponds to a prompt for a specific code, and each column corresponds to a question.

###### 1.2.2. Averaging Similarity Scores:
For each code (user code), we calculate the average similarity score across all prompts associated with that code. This is done by taking the mean along the rows of the similarity matrix.

###### 1.2.3. Storing Results:
The average similarity scores for each code are stored in the code2similarity dictionary, where the code serves as the key.

###### 1.2.4. Creating a DataFrame:
The similarity scores are organized into a DataFrame (similarity_df). Each row of the DataFrame corresponds to a question, and each column corresponds to a code. The values in the DataFrame represent the average cosine similarity score between the questions and the prompts for each code.

###### 1.2.5. Displaying Results:
The resulting DataFrame is printed, showing the average similarity scores between questions and prompts for each code


#### 1.3.

Next, we process the similarity DataFrame (similarity_df) obtained from the previous code section. It organizes and structures the similarity scores to create a new DataFrame (question_mapping_scores) that provides a mapping between codes and their respective similarity scores for each question. 

```
code2questionmapping = dict()
for code, cosine_scores in similarity_df.items():
    code2questionmapping[code] = similarity_df[code].tolist()

question_mapping_scores = pd.DataFrame(code2questionmapping).T
question_mapping_scores.reset_index(inplace=True)
question_mapping_scores.rename(columns={i: f"Q_{i}" for i in range(len(questions))}, inplace=True)
question_mapping_scores.rename(columns={"index" : "code"}, inplace=True)
question_mapping_scores
```
------------
------------
------------

## 2 - Feature Engineering 
To improve, we add following parts.

#### 2.1. 
We initialize a list of keywords (keywords2search)

#### 2.2. 
Then, this code cell processes conversations stored in code2convos and extracts various features related to user prompts and ChatGPT responses for each code. Additionally, it incorporates a pattern-based approach to identify if a user prompt contains specific error-related terms.

```
### Using pattern based approach in the structure of the sentences to tell if it is an error or not

code2features = defaultdict(lambda : defaultdict(int))

for code, convs in code2convos.items():
    if len(convs) == 0:
        print(code)
        continue
    for c in convs:
        text = c["text"].lower()
        if c["role"] == "user":
            # User Prompts
            # count the user prompts
            code2features[code]["#user_prompts"] += 1
            for kw in keywords2search:
                code2features[code][f"#{kw}"] +=  len(re.findall(rf"\b{kw}\b", text))

            code2features[code]["prompt_avg_chars"] += len(text)
        else:
            # ChatGPT Responses
            code2features[code]["response_avg_chars"] += len(text)

        code2features[code]["prompt_avg_chars"] /= code2features[code]["#user_prompts"]
        code2features[code]["response_avg_chars"] /= code2features[code]["#user_prompts"]
```

###### 2.2.1. Initialization
We initialize a defaultdict of defaultdict named code2features to store features for each code. This data structure is used to store counts related to user prompts and ChatGPT responses.

###### 2.2.2. Iterating Through Codes and Conversations:
Then it iterates through each code and its corresponding conversations (convs) in the code2convos dictionary.

###### 2.2.3. Counting User Prompts:
For each user prompt in the conversations, it increments the count of user prompts (#user_prompts) for the respective code.


###### 2.2.4. Counting Keyword Occurrences:
For each user prompt, we count the occurrences of keywords from the keywords2search list using regular expressions.

###### 2.2.5. Calculating Average Characters:
Keep track of the total number of characters in both user prompts (prompt_avg_chars) and ChatGPT responses (response_avg_chars). It later calculates the average characters for both.

###### 2.2.6. Printing Codes with No Conversations:
If there are no conversations (convs) for a particular code, it prints the code to the console.

###### 2.2.7. Normalization of Average Characters:
Finally we normalize the average characters for user prompts and ChatGPT responses by dividing the total characters by the number of user prompts.



#### 2.3.
Then we create a Pandas DataFrame (df) from the feature information stored in the code2features dictionary. 
```
df = pd.DataFrame(code2features).T
df.head(5)
```

#### 2.4. 
Next, we read a CSV file named "scores.csv" and store the resulting DataFrame in the variable named scores. Then some information about scores is displayed. 
```
scores = pd.read_csv("/content/scores.csv", sep=",")
scores["code"] = scores["code"].apply(lambda x: x.strip())

# selecting the columns we need and we care
scores = scores[["code", "grade"]]

# show some examples
scores.head()
```

#### 2.5. 
After that, we modify the structure of the Pandas DataFrame df that was created in a previous section. Reset_index method is used to reset the index of the DataFrame df. The  rename method is used to rename the column with the label "index" to "code". 
```
df.reset_index(inplace=True, drop=False)
df.rename(columns={"index": "code"}, inplace=True)
df.head()
```

#### 2.6. 
Then, we perform a left merge between two Pandas DataFrames (df and question_mapping_scores)
question_mapping_scores) is a dataframe that provides a mapping between codes and their respective similarity scores for each question, it's created earlier.
```
df = pd.merge(df, question_mapping_scores, on="code", how="left")
```

#### 2.7. 
Next, we extend the feature merging process by incorporating information from another DataFrame named scores. The result of the merge and data cleaning operations is stored back in the temp_df. 
```
# Merge the Features
temp_df = pd.merge(df, scores, on='code', how="left")
temp_df.dropna(inplace=True)
temp_df.drop_duplicates("code",inplace=True, keep="first")
temp_df.head()
```

#### 2.8. 
Then we set up the features (X) and the target variable (y) for a machine learning model.
```
# Set the features and target variables
X = temp_df[temp_df.columns[1:-1]]
y = temp_df["grade"]
print(X.shape, y.shape)
```
###### 2.8.1. Feature Selection
temp_df.columns[1:-1] selects all columns from the second column to the second-to-last column of the DataFrame (temp_df). These columns are considered as 			features for the machine learning model.
###### 2.8.2. Target Variable Selection
temp_df["grade"] selects the "grade" column from the DataFrame (temp_df). This column is considered as the target variable for the machine learning model.
###### 2.8.3. Setting Up Features and Target Variable
X is assigned the selected features.
y is assigned the selected target variable.


------------
------------
------------
## 3 - Apply feature subset selection algorithm
For this, releif algorithm is chosen as the feature subset selection.	

#### 3.1.
```
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
```
Here we use scikit-learn's train_test_split function to split the dataset into training and testing sets.

#### 3.2.
Here out code sets up and runs a genetic algorithm to find an optimal subset of features for a machine learning model
```
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from deap import base, creator, tools, algorithms
import random

# Assuming df is your DataFrame and the last column is the target variable
# Replace it with the actual target variable column name
target_column = 'grades'
X = X_train
y = y_train

# Create a fitness function
def evaluate(individual, X, y, clf):
    selected_features = [i for i, val in enumerate(individual) if val == 1]
    if not selected_features:
        return 0.0,  # Return a tuple
    X_subset = X.iloc[:, selected_features]
    scores = cross_val_score(clf, X_subset, y, cv=5, scoring='accuracy')
    return np.mean(scores),  # Return a tuple

# Set up the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
...
```
#### 3.2.
Then, we use the Information Gain filter method to select the top 10 features from the training set (X_train). The selected features are then printed to provide insights into which features are considered most informative for the classification task.
```
# Information Gain filter
info_gain_selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_info_gain = info_gain_selector.fit_transform(X_train, y_train)

print("\nInformation Gain Selected Features:")
print(X.columns[info_gain_selector.get_support()])
```
#### 3.3.
Next, we combine the features selected through the genetic algorithm (selected_from_genetic) and the features selected using the Information Gain filter method (selected_from_infogain). 
```
selected_from_genetic = ['#hyperparameter', '#correlation', '#adelie', '#variable', '#dataset', '#choose', '#chinstrap', '#import', 'Q_0', 'Q_4']
selected_from_infogain = ['#hyperparameter', '#adelie', '#variable', '#choose', 'Q_1', 'Q_2', 'Q_5', 'Q_8']
selected = (set(selected_from_genetic) | set(selected_from_infogain))
```

#### 3.4. 
Then, we select a subset of features from the training and testing sets based on the combined set of selected features obtained from both the genetic algorithm and the Information Gain filter.
```
#select best subsets according to selection
X_train_selected = X_train[selected]
X_test_selected = X_test[selected]

# Print the shapes of the resulting sets
print("Shape of X_train:", X_train_selected.shape)
print("Shape of X_test:", X_test_selected.shape)
```


## 4 - Model Building
We use linear regression with Lasso regression (skip feature subset selection in this case). Lasso regressison will provide another level of feature selection.

#### 4.1.
```
# Standardize features (important for Lasso regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a range of alpha values
alphas = np.arange(0.1, 3.1, 0.1)

# Lists to store results
mse_values = []
adjusted_r2_values = []

# Iterate over alpha values
for alpha in alphas:
    # Create and train Lasso regression model
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred_lasso = lasso_model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_lasso)
    r2 = r2_score(y_test, y_pred_lasso)

    # Calculate adjusted R-squared
    n = X_test_scaled.shape[0]
    k = X_test_scaled.shape[1]
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - k - 1))

    # Append results to lists
    mse_values.append(mse)
    adjusted_r2_values.append(adjusted_r2)
```

We try to do model building using Lasso regression with different values of the regularization parameter (alpha). We create 'alphas', an array of alpha values ranging from 0.1 to 3.0 with a step of 0.1. The code iterates over each alpha value in the alphas array.
In summary, this code performs Lasso regression with different regularization parameter values. It standardizes the features, iterates over a range of alpha values, builds a Lasso model for each alpha, evaluates the model on the testing set, and stores the MSE and adjusted R-squared values for each alpha in separate lists. These lists can be used to analyze the performance of the Lasso models with different levels of regularization.

#### 4.2. 
```
# Print alpha values along with their corresponding adjusted R2 and RMSE
print("Alpha\tAdjusted R2\tRMSE")
print("----------------------------------")

for i, alpha in enumerate(alphas):
    # Create and train Lasso regression model
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred_lasso = lasso_model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_lasso)
    r2 = r2_score(y_test, y_pred_lasso)

    # Calculate adjusted R-squared
    n = X_test_scaled.shape[0]
    k = X_test_scaled.shape[1]
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - k - 1))

    # Print alpha, adjusted R2, and RMSE
    print(f"{alpha:.2f}\t{adjusted_r2:.4f}\t\t{mse:.4f}")
```
Then we  print a table of alpha values along with their corresponding adjusted R-squared (adjusted R2) and Root Mean Squared Error (RMSE) values for Lasso regression. This code provides a clear output of the performance metrics (adjusted R2 and RMSE) for different alpha values in Lasso regression. This information is useful for selecting an appropriate alpha value that balances model complexity and performance on the testing set.
Finally, we plot the results.

# For CS412 Fall 2024 Term Project Version 2

## 1 - Prompt Matching

#### 1.1 Using BERT

```
# Step 1: Extract text from HTML files
html_files_path = '/content/dataset/*.html'
extracted_prompts = []

for html_file in tqdm(glob(html_files_path)):
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        texts = [p.get_text(strip=True) for p in soup.find_all('p')]
        extracted_prompts.extend(texts)

# Step 2: Extract homework questions from .ipynb file
notebook_path = '/content/Assignment.ipynb'
extracted_homework_questions = []

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown':
        if cell['source'].startswith('Question'):
            extracted_homework_questions.append(cell['source'])
```
Here we are extracting the HTML foles and the homework questions from the .ipynb file itself by using BeautifulSoup for the HTML files and only checking markup cells in the homework .ipynb file for the questions. Both the prompts and questions are placed in their respective list.</br>


```
# Function to encode text using BERT
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# If you're running this in a local setup and not in a hosted environment with a GPU,
# you should add the following line to tell PyTorch to use the CPU.
model = model.to('cpu')

# Step 3: Encode prompts and homework questions with BERT
prompts_embeddings = np.array([encode_text(prompt, tokenizer, model) for prompt in extracted_prompts])
questions_embeddings = np.array([encode_text(question, tokenizer, model) for question in extracted_homework_questions])

# Step 4: Calculate cosine similarity between prompts and homework questions
similarity_matrix = np.array([[1 - cosine(p, q) for q in questions_embeddings] for p in prompts_embeddings])

# Step 5: Find the best matching homework question for each prompt
for i, prompt in enumerate(extracted_prompts):
    best_match_idx = np.argmax(similarity_matrix[i])
    print(f"Prompt: {prompt}")
    print(f"Best matching homework question: {extracted_homework_questions[best_match_idx]}")
    print(f"Similarity score: {similarity_matrix[i][best_match_idx]}")
    print("\n")
```

Here we are using a tokenizer to encode the text we have. The tokenizer being used in BERT with an already pretrained model. </br>
Then we are turning the prompts and questions into np arrays, this allows us to encode the prompts and the questions. By doing so, we can calculate the cosine similarity between the prompts and the questions. The concept behing using it is the same as the first file. </br>
Finally we loop through the prompts and match them to the questions by finding the highest similarity.

#### 1.2 Using Sentence-Transformers

```
# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode texts to get sentence embeddings with batching
def encode_texts(texts, model, batch_size=32):
    embeddings = []
    for start_index in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        texts_batch = texts[start_index:start_index+batch_size]
        embeddings_batch = model.encode(texts_batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(embeddings_batch)
    return np.vstack(embeddings)

def extract_questions_from_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    extracted_questions = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            cell_text = cell['source']
            # Add the whole cell text as one question, removing any leading/trailing whitespace
            extracted_questions.append(cell_text.strip())
    return extracted_questions

# Extract text from HTML files
html_files_path = '/content/dataset/*.html'
extracted_prompts = []
for html_file in tqdm(glob(html_files_path), desc='Extracting prompts'):
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        texts = [p.get_text(strip=True) for p in soup.find_all('p')]
        extracted_prompts.extend(texts)

# Extract homework questions from .ipynb file
notebook_path = '/content/Assignment.ipynb'
extracted_homework_questions = extract_questions_from_notebook(notebook_path)

# Encode the prompts and homework questions
prompts_embeddings = encode_texts(extracted_prompts, model)
questions_embeddings = encode_texts(extracted_homework_questions, model)

# Calculate cosine similarity between prompts and homework questions
similarity_matrix = cosine_similarity(prompts_embeddings, questions_embeddings)

# Find the best matching homework question for each prompt
for i, prompt_embedding in enumerate(prompts_embeddings):
    best_match_idx = np.argmax(similarity_matrix[i])
    print(f"Prompt: {extracted_prompts[i]}")
    print(f"Best matching homework question: {extracted_homework_questions[best_match_idx]}")
    print(f"Similarity score: {similarity_matrix[i][best_match_idx]}")
    print("\n")
```

Here we are using a less computationally expensive model to perform the same task above. The difference is that instead of using a tokenizer, we are using sentence embeddings with batching. For this reason, the code is similar except where the transformer was used.

### Contributions

Fatma Khalil (29999): Prompt matching and debugging </br>
Muhammed Abdulrahman (30043): Feature Extraction and model training </br>
Taneera Tajammul (29572): Experimenting and Model testing </br>
Emir Asal (27933): Features and correlations, improving the model and bug fixing </br>
Selman Sezer (29572): Research and Readme file </br>
