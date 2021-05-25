# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import math
import numpy as np
from porter_stemmer import PorterStemmer


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.creative = creative
        self.clarifying = False
        self.recommendation_made = False

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        # numpy array
        self.user_ratings = np.zeros(len(self.titles))
        self.recommendations = []
        self.recommendation_counter = 0
        self.input_counter = 0

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def prompt_for_info(self):
        return 'Tell me about another movie you have seen.'

    def echo_sentiment(self, sentiment, title):
        phrase = ''
        if sentiment > 0:
            phrase = 'You liked'
        else:
            phrase = 'You did not like'
        return phrase + ' "' + title + '". Thank you!\n'
    
    def other_response(self, line):
        return "Hm, I'm not quite sure what you mean. Why don't you tell me about another movie?"
    
    def prompt_for_clarification(self, title_ids):
        movies = []
        for id in title_ids:
            movies.append(self.titles[id][0])
        return 'I found multiple results for your input. Which movie did you mean?' + movies

    def unclear_sentiment_response(self, movie):
        return "I'm sorry, I'm not quite sure if you liked \"" + movie + "\". \n Tell me more about \"" + movie + '".'
        


    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            response = "I processed {} in starter mode!!".format(line)

        # CHECK IF THE USER WANTS TO HEAR ANOTHER RECOMMENDATION
        if self.recommendation_made:
            lower_response = line.lower()
            affirmations = ['yes', 'yeah', 'yea', 'ya', 'y', 'sure', 'okay', 'ok']
            if lower_response in affirmations:
                return self.recommend_movie()
        
        # CHECK IF THE RESPONSE WAS A CLARIFICATION
        if self.clarifying:
            self.clarifying = False 
            title_ids = disambiguate(line, title_ids)
            
        input_titles = self.extract_titles(line)
        sentiments = []
        sentiment = 0

        # NO TITLES FOUND
        if len(input_titles) == 0:
            return "Sorry, I don't understand. Tell me about a movie that you have seen."

        # MORE THAN ONE TITLE FOUND IN CREATIVE MODE --> EXTRACT SENTIMENT FOR MULTIPLE MOVIES
        if self.creative and len(input_titles) > 1:
            sentiments = extract_sentiment_for_movies(line)
        
        # MORE THAN ONE TITLE FOUND IN NON-CREATIVE MODE
        if not self.creative and len(input_titles) > 1:
            return "Please tell me about one movie at a time. Go ahead."
        
        # FIND SENTIMENT OF SINGLE MOVIE IN NON-CREATIVE MODE
        if not self.creative:
            sentiment = self.extract_sentiment(line)
            if sentiment == 0:
                return unclear_sentiment_response(input_titles[0])
            
        # CHECK IF SENTIMENTS WERE FOUND IN CREATIVE MODE
        for i in range(len(sentiments)):
            if sentiments[i] == 0:
                return unclear_sentiment_response(sentiments[i][0])

        title_ids = []
        # GRAB TITLE IDS FOR A SINGLE MOVIE
        if not self.creative or len(input_titles) == 1:
            title_ids = self.find_movies_by_title(input_titles[0])
        # GRAB TITLE IDS FOR MOVIES IN CREATIVE MODE
        else:
             title_ids = self.find_movies_by_title(input_titles)

        # MULTIPLE TITLE-IDS FOUND IN NON-CREATIVE MODE
        if len(title_ids) > 1 and not self.creative:
            return 'Sorry, I cannot find the requested movie. Can you be more specific?'

        # MULTIPLE TITLE-IDS FOUND IN CREATIVE MODE
        if len(title_ids) > 1 and self.creative:
            self.clarifying = True
            return prompt_for_clarification(title_ids)
        
        # UPDATE USER RATINGS IN NON-CREATIVE MODE
        if not self.creative:
            self.user_ratings[title_ids[0]] = sentiment
            self.input_counter += 1
        # UPDATE USER RATINGS IN CREATIVE MODE
        else:
            for i in range(len(title_ids)):
                self.user_ratings[title_ids[i]] = sentiments[i][1]
                self.input_counter += 1
        
        if self.input_counter < 5:
            if not self.creative or len(title_ids) == 1:
                return self.echo_sentiment(sentiment, input_titles[0]) + self.prompt_for_info()
            else:
                # RESPONSE FOR ACKNOWLEDGING MULTIPLE MOVIES AND HOW THE USER RATED THEM
        else:
            self.recommendations.extend(self.recommend(np.array(self.user_ratings), self.ratings))
            return "That's enough for me to make a recommendation.\n" + self.recommend_movie()
        




        '''
        if self.clarifying:
            self.clarifying = False 
            title_id = self.disambiguate(line, title_ids)
        if line[0].lower() == 'y' and self.input_counter >= 5:
            return self.recommend_movie()
        input_titles = self.extract_titles(line)
<<<<<<< HEAD
        if len(input_titles) == 0:
            return "Sorry, I don't understand. Tell me about a movie that you have seen."
        if len(input_titles) > 1 and not self.creative:
=======
        if len(input_titles) == 0 and not self.creative:
           return "Sorry, I don't understand. Tell me about a movie that you have seen."
        if len(input_titles) == 0 and self.creative:
            #why are we doing something other than this in extract_titles when self.creative is off ? 
            input_titles = re.findall(r'"([^"]*)"', line)
            return self.spellcheck(input_titles, 3)
        if len(input_titles) > 1:
>>>>>>> 721afa44d135dab1b51f8e3b631f96753f8de80f
            return "Please tell me about one movie at a time. Go ahead."
        # handle case of when they are updating their rating (want to do -=1 for counter)
        input_sentiment = self.extract_sentiment(line)
        if input_sentiment == 0:
            return "I'm sorry, I'm not quite sure if you liked \"" + input_titles[0] + "\". \n Tell me more about \"" + input_titles[0] + '".'
        # need to change to work for all lower
        title_ids = self.find_movies_by_title(input_titles[0])
        '''
        #for i in range(len(title_ids)):
         #   self.user_ratings[title_ids[i]] = input_sentiment
        '''
        if self.creative and len(title_ids) > 1:
            self.clarifying = True
            return self.prompt_for_clarification(title_ids)
        self.input_counter += 1
        if self.input_counter < 5:
            # prompt user for more info
            return self.echo_sentiment(input_sentiment, input_titles[0]) + self.prompt_for_info()
        else:
            self.recommendations.extend(self.recommend(
                np.array(self.user_ratings), self.ratings))
            return "That's enough for me to make a recommendation.\n" + self.recommend_movie()
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response
        '''

    def recommend_movie(self):
        # check if there are any more recommendations
        if self.recommendation_counter < len(self.recommendations):
            recommended_movie = self.recommendations[self.recommendation_counter]
            recommended_movie = self.titles[recommended_movie][0]
            self.recommendation_counter += 1
            self.recommendation_made = True
            return 'u wld like ' + recommended_movie + '. wld u like to hear another recommendation?'
        else:
            self.recommendation_made = False
            return self.prompt_for_info()
        ""

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        input_titles = re.findall(r'"([^"]*)"', preprocessed_input)
        if self.creative is False or len(input_titles) != 0:
            return input_titles

        def rearrange(w, title):
            if title.find("(") == -1:
                return title.split(w, 1)[1] + ", " + w
            title = title.split(w,1)[1]
            year_index = title.find("(")
            return title[:year_index-1] + ", "+ w + title[year_index:]

        def check_titles(title):
            if title.find('the ') == 0:
                title = rearrange('the ',title)
            if title.find('an ') == 0:
                title = rearrange("an ", title)
            for id in range(len(self.titles)):
                db_title = self.titles[id][0].lower().split("(")[0]
                if title == db_title.strip():
                    return True

        tokens = preprocessed_input.lower().split()
        size = 1
        res = [] 

        while size < len(tokens):    
            for i in range(len(tokens)):
                substr = " ".join(tokens[i:i+size])
                if size == 1 and tokens[i] in ['i', 'a']:
                    continue

                if check_titles(substr) and substr not in res:
                        substr.capitalize()
                        res.append(substr)
            size += 1
        

        return [max(res)]

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        # Rearrange input title to match database entries
        # Changes the input title to resemble the database entry: <Title>, *An or The* (Year)
        # ex. The Notebook (2004) -> Notebook, The (2004) 
        def rearrange(w, title):
            if title.find("(") == -1:
                return (title.split(w, 1)[1] + ", " + w).strip()
            title = title.split(w,1)[1]
            year_index = title.find("(")
            return title[:year_index-1] + ", "+ w + title[year_index:]
        
        # Compare the year of two titles (input, database) to see if they match. 
        # if no year was given for the input title, then ignore this check.
        def compare_years(t1,t2):
            t1_y = re.search(r'\(([0-9]+)\)', t1)
            if t1_y == None:
                s_t1, s_t2 = t1.lower().strip(), t2.lower().split("(")[0].strip()
                if s_t1 in s_t2:
                    return s_t1.split()[0] == s_t2.split()[0]
                else:
                    alt_t2 = re.search(r'\(([\w][\D][^(]+)\)',t2)
                    if alt_t2 is not None:
                        return t1.lower().strip() in alt_t2.group(1).lower().strip()                    
                    return False
                    
            t2_y = re.search(r'\(([0-9]+)\)', t2)
            return t1_y.group(1) == t2_y.group(1)

        # Handle the case that an input title begins with "An" or "The" 
        for w in ['The ', 'An ', 'La ', 'Les ', 'Le ', 'L ']:
            if title.find(w) == 0:
                title = rearrange(w, title)

        ids = []    
        # Iterate through databse and add matching movies to the resulting array
        for id in range(len(self.titles)):
            if title.lower() in self.titles[id][0].lower():
                print(self.titles[id][0])
                if compare_years(title, self.titles[id][0]): 
                    ids.append(id)
        
        return ids

    def extract_sentiment(self, preprocessed_input):
        
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        stemmer = PorterStemmer()
        split_input = preprocessed_input.lower().split()
        negate = 1
        count = 0
        in_quotes = False
        power = 1
        neg_list = ["no", "not", "rather", "couldn't", "wasn't", "didn't", "wouldn't", "shouldn't", "weren't", "don't", "doesn't", "haven't", "hasn't", "won't", "wont", "hadn't", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "isn't", "can't", "cannot", "mustn't", "mightn't", "shan't", "without", "needn't"]
        power_list = ["really", "reeally", "loved", "love", "hate", "hated", "terrible", "amazing", "fantastic", "incredible", "dreadful", "horrible", "horrid", "horrendous"]
        for word in split_input:
            word = word.strip()
            word_no_comma = word.rstrip(",")
            stem = stemmer.stem(word_no_comma, 0, len(word_no_comma) - 1)
            if stem.endswith('i'):
                stem = stem[:-1] + 'y'
            if word.startswith("\""):
                in_quotes = True
            #if word.endswith("\"") or "\"" in word:
            if word.endswith("\""):
                in_quotes = False
                continue
            if in_quotes:
                continue
            if word in neg_list and not word.endswith(","): # if word in neg_list but ends in comma, negate would be positive
                negate = -1  # or have negate * -1
            else:
                has_comma = False
                # maybe include other punctuation? 
                if word.endswith(","):
                    has_comma = True
                if self.creative:
                    if word_no_comma in power_list or stem in power_list or word.endswith("!"):
                        power = 2
                if word_no_comma in self.sentiment:
                    if self.sentiment[word_no_comma] == "pos":
                        count += 1 * negate
                    else:
                        count += -1 * negate
                elif stem in self.sentiment:
                    if self.sentiment[stem] == "pos":
                        count += 1 * negate
                    else:
                        count += -1 * negate  
                if has_comma:
                    negate = 1
        print(count)
        print(power)
        print(negate)
        if count > 0:
            return 1 * power
        elif count < 0:
            return -1 * power
        return 0

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        pass

    #Helper function for spellchecking user input
    def spellcheck(self,titles, max_distance):
        potential_movies = self.find_movies_closest_to_title(titles[0], max_distance)
        
        if(len(potential_movies) != 0):
            index = potential_movies[0]
            return "Did you mean " + self.titles[index][0] + " ?"

        return "Sorry, I am unfamiliar with that movie. Are you sure you're spelling it correctly?"


    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """

        # get length of title N
        # for every other title 
            # get length of title M

            #create an empty numpy array of shape NM 
            #array[i][0] = i 
            #array[0][j] = j 

                # for i ... n 
                    #for j ... m 
                        #array[i][j] = min 
                        #min of array[i-1][j] + 1
                        #or array[i][j-1] + 1
                        #or array[i-1][j-1] + 2 if title[i] \neq other_title[j] but 0 otherwise 

        
        #get titles from self.titles
        #titles = ["asdfgh", "light", "drk"]

        # Rearrange input title to match database entries
        # Changes the input title to resemble the database entry: <Title>, *An or The* (Year)
        # ex. The Notebook (2004) -> Notebook, The (2004) 
        def rearrange(w, title):
            if title.find("(") == -1:
                return (title.split(w, 1)[1] + ", " + w).strip()
            title = title.split(w,1)[1]
            year_index = title.find("(")
            return title[:year_index-1] + ", "+ w + title[year_index:]
        
        # Handle the case that an input title begins with "An" or "The" 
        for w in ['The ', 'An ', 'La ', 'Les ', 'Le ', 'L ',
        'the', 'an', 'la', 'les', 'le', 'l'
        ]:
            if title.find(w) == 0:
                title = rearrange(w, title)

        titles = self.titles
        title = title.strip().lower()
        length_first = len(title)
        title_rev =  title[::-1]
        distances = []
     
        for i in range(len(titles)):
          #access index 1 of the other_title field to get the name
            other_title = titles[i][0]
            length_orig = len(other_title)
            other_title = other_title[0:length_orig-7].strip().lower()
            length_second = len(other_title)
            

            index = i
          # get index of movie

            arr= np.zeros((length_first+1, length_second+1))
            
            for i in range(length_first+1):
                arr[i][0]= length_first - i 

            for i in range(length_second+1):
                arr[length_first][i]= i 


            for i in range(length_first-1, -1, -1): 
                for j in range(1,length_second+1):
                    left = arr[i][j-1] + 1
                    bottom = arr[i+1][j] + 1
                    diagonal = arr[i+1][j-1]


                    if title_rev[i] != other_title[j-1]:
                        diagonal += 2

                    arr[i][j] = min(left, bottom, diagonal)
            

            
            distance = arr[0][length_second]

            if(distance <= max_distance):
                distance_betweeen = (distance, other_title,index)
                distances.append(distance_betweeen)

        distances = sorted(distances, key = lambda x: x[0])


        if (len(distances) != 0):
            minimum = distances[0][0]
            final_list = [x[2] for x in distances if x[0] == minimum]
            return final_list

        else: 
            return []



    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        ids = []
        for candidate in candidates:
<<<<<<< HEAD
            if clarification == self.titles[candidate][0]:
                #ids.append(candidate)
                return candidate
=======
            # Check without year if year not given 
            # Check with year if year is given
            title = self.titles[candidate][0]
            year = re.search(r'([\d]{4})', title)
            if year is not None and year.group(0) in clarification:
                ids.append(candidate)
                continue
            if clarification in title.split("(")[0]:
                ids.append(candidate)
>>>>>>> 721afa44d135dab1b51f8e3b631f96753f8de80f
        return ids

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        for i in range(len(ratings)):
          movie = ratings[i]
          for user in range(len(movie)):
              if movie[user] == 0:
                continue 
              ratings[i][user]= 1 if movie[user] > threshold else -1
        binarized_ratings = ratings

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        # find the indices that are common aka don't have 0 
        u_sum = 0
        v_sum = 0
        numerator = 0
        for i in range(len(u)):
            if u[i] != 0 and v[i] != 0:
                u_sum += (u[i] ** 2)
                v_sum += (v[i] ** 2)
                numerator += (u[i] * v[i])
        if u_sum == 0 or v_sum == 0:
            return 0
        return numerator / (math.sqrt(u_sum) * math.sqrt(v_sum))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        similarities = np.zeros((len(self.ratings), len(self.ratings)))
        recommendations = []
        # compute all the similarities
        for i in range(len(ratings_matrix)):
            if user_ratings[i] == 0:
                continue
            for j in range(len(ratings_matrix)):
                #if i != j:
                cos_sim = self.similarity(ratings_matrix[i], ratings_matrix[j])
                similarities[i][j] = cos_sim
        # find the user's projected rating for each movie 
        projected_ratings = []
        for i in range(len(ratings_matrix)):
            # want to calculate projected ratings for each movie that the user has not yet seen
            #if user_ratings[i] != 0:
                #continue
            projected_rating = 0
            for j in range(len(ratings_matrix)):
                # compare the movie to every other movie
                # check that there is a similarity score computed + that the user has rated movie j
                #if i == j or user_ratings[i] != 0 or user_ratings[j] == 0:
                    #continue      
                projected_rating += (similarities[i][j] * user_ratings[j])
            projected_ratings.append((projected_rating, i))
            # need to also keep track of the indices of the movies
            # do tuples, and sort by the rating, but then insert the indices into the index 
        projected_ratings.sort(key=lambda tup: tup[0], reverse=True)
        print(projected_ratings)
        for i in range(k):
            recommendations.append(projected_ratings[i][1])
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
