# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import math
import numpy as np


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        #numpy array
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
        return 'please tell me more'

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
        # if the user says yes and dict is large enough, supply a recommendation
        #if line == 'Yes' or line == 'yes' or line == 'Yeah' or line == 'yeah':
        if line[0].lower() == 'y':
            self.recommend_movie()
        input_titles = self.extract_titles(line)
        if len(input_titles) == 0:
            return 'response that title was not found / only supply one' 
        input_sentiment = self.extract_sentiment(line)
        if input_sentiment == 0:
            return "response that the chat-bot doesn't know how they feel about the movie"
        # need to change to work for all lower
        title_indices = self.find_movies_by_title(input_titles[0])
        print(title_indices)
        if len(title_indices) > 1 or len(title_indices) == 0:
            return 'response for the user to clarify what movie' 
        self.user_ratings[title_indices[0]] = input_sentiment
        self.input_counter += 1
        if self.input_counter < 5:
            # prompt user for more info
            return self.prompt_for_info()
        else:
            # do some array magic

            self.recommendations.extend(self.recommend(np.array(self.user_ratings), self.ratings))
            self.recommend_movie()
            # read first recommendation to the user
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    def recommend_movie(self):
        # check if there are any more recommendations 
        if self.recommendation_counter < len(self.recommendations):
            recommended_movie = self.recommendations[recommendation_counter]
            return 'u wld like' + recommended_movie + '. wld u like to hear another recommendation?'
        else:
            # prompt for more info 
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
        if self.creative is False:
            input_titles = re.findall(r'"([^"]*)"', preprocessed_input)
            return input_titles

        return []

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
                return title.split(w, 1)[1] + ", " + w
            title = title.split(w,1)[1]
            year_index = title.find("(")
            return title[:year_index-1] + ", "+ w + title[year_index:]
        
        # Compare the year of two titles (input, database) to see if they match. 
        # if no year was given for the input title, then ignore this check.
        def compare_years(t1,t2):
            t1_y = re.search(r'\(([0-9]+)\)', t1)
            if t1_y == None:
                return True
            t2_y = re.search(r'\(([0-9]+)\)', t2)
            return t1_y.group(0) == t2_y.group(0)

        # Handle the case that an input title begins with "An" or "The" 
        if title.find('The ') == 0:
            title = rearrange('The ',title)
        if title.find('An ') == 0:
            title = rearrange("An ", title)

        ids = []    
        # Iterate through databse and add matching movies to the resulting array
        for id in range(len(self.titles)):
            if title in self.titles[id][0]:
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
        split_input = preprocessed_input.lower().split()
        negate = 1
        pos_count = 0
        neg_count = 0
        neg_list = ["no", "not", "rather", "couldn’t", "wasn’t", "didn’t", "wouldn’t", "shouldn’t", "weren’t", "don’t", "doesn’t", "haven’t", "hasn’t", "won’t", "wont", "hadn’t", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "isn’t", "can’t", "cannot", "mustn’t", "mightn’t", "shan’t", "without", "needn’t"]
        for word in split_input:
            if word in neg_list: # if word in neg_list but ends in comma, negate would be positive again so no need to strip comma here
                negate = -1  # or have negate * -1
            else:
                has_comma = False
                # maybe include other punctuation? 
                if word.endswith(","):
                    has_comma = True
                    word = word.rstrip(",")
                if word in self.sentiment:
                    if self.sentiment[word] == "pos":
                        pos_count += (1 * negate)
                    else:
                        neg_count += (1 * negate)
                if has_comma:
                    negate = 1
        if pos_count > neg_count:
            return 1
        elif pos_count < neg_count:
            return -1
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

        pass

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
        pass

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
        print('COMPUTING SIMILARITIES')
        for i in range(len(ratings_matrix)):
            if user_ratings[i] == 0:
                continue
            for j in range(len(ratings_matrix)):
                if i != j:
                    cos_sim = self.similarity(ratings_matrix[i], ratings_matrix[j])
                    similarities[i][j] = cos_sim
        # find the user's projected rating for each movie 
        projected_ratings = []
        print('FINDING RATINGS!')
        for i in range(len(ratings_matrix)):
            projected_rating = 0
            for j in range(len(ratings_matrix)):
                if i == j or user_ratings[i] != 0 or user_ratings[j] == 0:
                    continue
                projected_rating += (similarities[i][j] * user_ratings[j])
            projected_ratings.append(projected_rating)
        projected_ratings = projected_ratings.sort(reverse=True)
        for i in range(k):
            recommendations.append(projected_ratings[i])
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
