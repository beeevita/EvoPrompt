
v1 = {
    "sim":"""Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt.
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
Different parts:
"input text" vs "my complex sentence"
"simpler text" vs "simpler terms, but keep the meaning"

2. Randomly mutate the different parts:
"input text" -> "provided text"
"my complex sentence" -> "the difficult sentence"
"simpler text" -> "easier language"
"simpler terms, but keep the meaning" -> "simpler words while maintaining the meaning"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.
New Prompt: Transform the provided text into easier language while maintaining the meaning, making it accessible for non-native English speakers.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.
Final Prompt: <prompt>Convert the difficult sentence into simpler words while preserving the meaning, so it's easier for non-native English speakers to understand.</prompt>


Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step2 and generate a new prompt.
Prompt 3: <prompt3>
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: <prompt0>

1. """,
"cls":{
    "sst-5":"""Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt.
Prompt 3: Assess a movie or a book based on its explanation and determine the sentiment of the movie review. Have your colleague's evaluation of the movie they watched be expressed in a concise remark (e.g. awesome, all right, terrible, or horrendous) following the narrative synopsis they were provided, and choose from terrible, bad, okay, good and great to describe the movie.
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
Different parts:
"classify the comment" vs "classify a sentence"
"Your task is to" vs "In this task, you are given sentences from movie reviews. The task is to"

2. Randomly mutate the different parts:
"classify the comment" -> "categorize the statement"
"classify a sentence" -> "evaluate the review"
"Your task is to" -> "Your mission is to"
"In this task, you are given sentences from movie reviews. The task is to" -> "In this assignment, you will receive movie review sentences. Your job is to"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Assess a movie or a book based on its explanation and determine the sentiment of the movie review. Have your colleague's evaluation of the movie they watched be expressed in a concise remark (e.g. awesome, all right, terrible, or horrendous) following the narrative synopsis they were provided, and choose from terrible, bad, okay, good and great to describe the movie.
New Prompt: In this assignment, you will receive movie review sentences. Your job is to evaluate the review and determine the sentiment, choosing from terrible, bad, okay, good, and great to describe the movie.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.
Final Prompt: <prompt>Your mission is to categorize the statement from a movie review by understanding its meaning and context, and then classify it as one of the following categories: terrible, bad, okay, good, or great.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step2 and generate a new prompt.
Prompt 3: <prompt3>
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: <prompt0>

1. """
}
}

v2 = {
    "sim":"""Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
Different parts:
"input text" vs "my complex sentence"
"simpler text" vs "simpler terms, but keep the meaning"

2. Randomly mutate the different parts:
"input text" -> "provided text"
"my complex sentence" -> "the difficult sentence"
"simpler text" -> "easier language"
"simpler terms, but keep the meaning" -> "simpler words while maintaining the meaning"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.

Final Prompt: <prompt>Transform the difficult sentence into easier language while keeping the meaning, for non-native English speakers to comprehend.</prompt>


Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1. """,
"cls":{
    "sst-5":"""Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
Different parts:
"Your task is to classify the comment" vs "In this task, you are given sentences from movie reviews. The task is to classify a sentence"
"comment" vs "sentences from movie reviews"

2. Randomly mutate the different parts:
"Your task is to classify the comment" -> "The objective is to categorize the statement"
"comment" -> "phrases in movie reviews"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.

Final Prompt: <prompt>As a sentiment classifier, analyze phrases in movie reviews and categorize them into one of the following categories: terrible, bad, okay, good, great, while considering the meaning and relevant context.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1. """
}
}

v3 = {
    "sim":"""Please follow the instruction step-by-step to generate a better prompt.    
1. Randomly mutate Prompt 1 and Prompt 2    
Prompt 1: Rewrite the input text into simpler text.    
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.    
2. Combine mutated prompts generated in step1 with the following Prompt 3 to generate a new prompt.    
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.    
3. Crossover the prompt generated in the step2 and the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:    
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.

1. Randomly mutate Prompt 1 and Prompt 2   
Mutated Prompt 1: Transform the original text into a simpler version.
Mutated Prompt 2: Simplify my complex sentence while maintaining its meaning.
2. Combine mutated prompts generated in step 1 with the following Prompt 3 to generate a new prompt.
New Combined Prompt: Transform the provided input text into simpler English sentences, maintaining the same meaning, to make it understood by non-native English speakers.
3. Crossover the prompt generated in step 2 and the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Crossover Final Prompt: <prompt> Simplify the provided sentence to make it easier for non-native English speakers to understand, maintaining its meaning. </prompt>

Please follow the instruction step-by-step to generate a better prompt.    
1. Randomly mutate Prompt 1 and Prompt 2    
Prompt 1: <prompt1>
Prompt 2: <prompt2>  
2. Combine mutated prompts generated in step1 with the following Prompt 3 to generate a new prompt.    
Prompt 3: <prompt3>  
3. Crossover the prompt generated in the step2 and the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:    
Basic Prompt: <prompt0>

1. """,
"cls":{
"sst-5":"""Please follow the instruction step-by-step to generate a better prompt.        
1. Randomly mutate Prompt 1 and Prompt 2        
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.    
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.    
2. Combine mutated prompts generated in step1 with the following Prompt 3 to generate a new prompt.        
Prompt 3: Prompt 3: Assess a movie or a book based on its explanation and determine the sentiment of the movie review. Have your colleague's evaluation of the movie they watched be expressed in a concise remark (e.g. awesome, all right, terrible, or horrendous) following the narrative synopsis they were provided, and choose from terrible, bad, okay, good and great to describe the movie.    
3. Crossover the prompt generated in the step2 and the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:        
Basic Prompt: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.    

1. Randomly mutate Prompt 1 and Prompt 2
Mutated Prompt 1: Your goal is to categorize the given comment as terrible, bad, okay, good, or great.
Mutated Prompt 2: You will be provided with sentences from movie reviews, and you need to classify them into one of these categories: terrible, bad, okay, good, great.
2. Combine mutated prompts generated in step 1 with the following Prompt 3 to generate a new prompt.
New Combined Prompt: Analyze the given movie or book review and categorize the sentiment of the review into one of these categories: terrible, bad, okay, good, or great, based on the explanation provided.
3. Crossover the prompt generated in step 2 and the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Crossover Final Prompt: <prompt> As a sentiment classifier, understand the meaning of the provided sentence and relevant context, then categorize it as terrible, bad, okay, good, or great. </prompt>,

Please follow the instruction step-by-step to generate a better prompt.    
1. Randomly mutate Prompt 1 and Prompt 2    
Prompt 1: <prompt1>
Prompt 2: <prompt2>  
2. Combine mutated prompts generated in step1 with the following Prompt 3 to generate a new prompt.    
Prompt 3: <prompt3>  
3. Crossover the prompt generated in the step2 and the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:    
Basic Prompt: <prompt0>

1. """}
}