# %%
from typing import Optional, Union
import pandas as pd
import numpy as np
import requests, re, os

from concurrent.futures import ProcessPoolExecutor

from .cypher import Cypher

class Decrypter:
    """Simple MCMC decryption algorithm using random walk"""
    def __init__(self, cypher_class: Cypher,
                       reference_text_url: str, 
                       allowable_chars: str = r"[a-z0-9\s]", 
                       allow_uppercase:bool = False,
                       max_workers: Optional[int] = None,
                       **kwargs) -> None:

        self.cypher_class           = cypher_class
        self.reference_text_url     = reference_text_url
        allow_tuple                 = self.generate_allowable_chars(allowable_chars)
        self.allowable_chars        = allow_tuple[0]
        self.allowable_char_pattern = allow_tuple[1]
        self.allow_uppercase        = allow_uppercase
        self.max_workers            = max_workers if max_workers is not None else os.cpu_count()
        self.kwargs                 = kwargs
        
        self.reference_text         = self.get_text()
        self.reference_text         = self.clean_text(self.reference_text)
    
    def generate_allowable_chars(self, pattern:str) -> tuple[list[str], str]:
        """Generates a list of allowable characters based on the given pattern

        Args:
            pattern (str): pattern to generate allowable characters from

        Returns:
            list: allowable characters
        """
        textset         = "".join([chr(i) for i in range(32, 127)])
        #If the pattern provided does not contain square brackets, assume that this is a list of characters. Generate a regex pattern from the list
        #Otherwise, assume that the pattern is a regex pattern.
        if "[" not in pattern:
            #Sort the characters in the pattern to ensure that the order is consistent and replace key characters with their escaped versions
            pattern = "".join([re.escape(char) for char in sorted(set(pattern))])
            pattern = "[" + pattern + "]"
        
        #Generate a list of allowable characters based on the given or generated pattern
        return re.findall(pattern, textset), pattern

    def get_text(self) -> str:
        """Retrieves text from given url

        Returns:
            str: text from url
        """
        r = requests.get(self.reference_text_url)
        return r.text
    
    def clean_text(self, text:str) -> str:
        """Removes non-allowable characters from text

        Args:
            text (str): text to clean

        Returns:
            str: cleaned text
        """

        #Find the substring that marks the end of the text
        start_substring = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK ", text).end()
        end_substring   = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK ", text).start()
        text            = text[start_substring:end_substring]

        #If uppercase are not allowed, turn them into lowercase
        if not self.allow_uppercase:
            text = text.lower()
        
        #Remove non-allowable characters        
        text = re.sub(self.invert_regex_pattern(), "", text)

        #Convert all whitespace to single space
        text = re.sub(r"\s+", " ", text)
        
        return text
    
    def invert_regex_pattern(self, pattern: Optional[str] = None) -> str:
        """Inverts the given regex pattern. Only works with simple regex patterns

        Args:
            pattern (Optional[str], optional): Simple regex pattern to invert. Defaults to None.

        Returns:
            str: inverted regex pattern
        """
        if pattern is None:
            pattern = self.allowable_char_pattern
        
        pattern = re.sub(r"\[", r"[^", pattern)
        return pattern
    
    def generate_transition_matrix(self, text: str, initialized_dataframe:pd.DataFrame) -> pd.DataFrame:
        """Generates transition matrix from text

        Args:
            text (str): text to generate transition matrix from

        Returns:
            pd.DataFrame: transition matrix
        """
        
        #Iterate through each character in the text.
        for i in range(len(text) - 1):
            initialized_dataframe.loc[text[i], text[i+1]] += 1
        
        return initialized_dataframe
    
    def get_charset(self, text:str, sorted:bool = False) -> Union[set[str], list[str]]:
        """Gets the charset of the given text

        Args:
            text (str): text to get charset for

        Returns:
            set: charset
        """
        return_set = set(text)
        if sorted:
            return_set = list(return_set)
            return_set.sort()
        
        return return_set
    
    def generate_transition_matrix_from_text(self) -> None:
        """Generates transition matrix from text

        Returns:
            pd.DataFrame: transition matrix
        """
        #Check if the transition matrix has already been generated
        if hasattr(self, "transition_matrix"):
            return
        
        #Generate a set of all characters in the text then sort them
        self.charset    = self.get_charset(self.reference_text, sorted=True)

        #Create a dataframe with the characters as the columns and the characters as the rows
        transition_matrix = pd.DataFrame(0, index=self.charset, columns=self.charset)

        #Begin preparations for multiprocessing to speed up the process

        #Create a list of the text to be split up into chunks
        chunk_length = int(np.ceil(len(self.reference_text) / self.max_workers))

        text_chunks = [
            self.reference_text[i * chunk_length : (i + 1)  * chunk_length] 
            for i in range(self.max_workers)
        ]

        #Create a list of the dataframes to be used for multiprocessing
        dataframes = [transition_matrix.copy() for _ in range(self.max_workers)]

        #Start multiprocessing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            dataframe_list = executor.map(self.generate_transition_matrix, text_chunks, dataframes)
        
        #Combine the dataframes
        for dataframe in dataframe_list:
            transition_matrix += dataframe
        
        #Add the transitions between the chunks
        for i in range(self.max_workers - 1):
            transition_matrix.loc[text_chunks[i][-1], text_chunks[i+1][0]] += 1
        
        self.transition_matrix = transition_matrix

        #Normalize the transition matrix so that each row sums to 1 and replace nan values with 0
        self.transition_matrix = self.transition_matrix.div(self.transition_matrix.sum(axis=1), axis=0).fillna(0)

        #Take the log of the transition matrix with a hard bottom of the lowest non-zero value in the matrix minus 20
        transition_matrix       = self.transition_matrix.to_numpy()
        nonzero_values          = transition_matrix[transition_matrix > 0]
        min_value               = np.log(np.min(nonzero_values))
        self.transition_matrix  = np.log(self.transition_matrix).clip(min_value - 20)
        
        return
    
    def find_cypher(self, text:str, max_iterations:int = 10_000) -> dict:
        """Finds the cypher for the given text

        Args:
            text (str): text to find cypher for
            max_iterations (int, optional): Maximum number of iterations to run. Defaults to 100.
            max_plausibility (float, optional): Maximum plausibility of the cypher. Defaults to 0.99.

        Returns:
            dict: cypher
        """
        self.text = text
        self.generate_transition_matrix_from_text()
        
        #Generate a cypher, then calculate its plausibility
        cypher              = self.cypher_class.generate_cypher(self.allowable_chars)
        cypher.plausibility = self.calculate_plausibility(cypher)

        #Keep track of the best cypher found so far as per the plausibility. This will be the cypher that is returned
        self.best_cypher = cypher
        
        for i in range(max_iterations):
            #Generate the cypher based on the current cypher and calculate its plausibility
            new_cypher              = cypher.update(cypher)
            new_cypher.plausibility = self.calculate_plausibility(new_cypher)

            #If the new cypher is better than the current best cypher, replace it
            if new_cypher.plausibility >= cypher.plausibility:
                cypher = new_cypher
                #If it's also better than the best cypher found so far, replace that too
                if cypher.plausibility >= self.best_cypher.plausibility:
                    self.best_cypher = cypher
            else:
                #If the new cypher is worse than the current cypher, flip a coin with acceptance P(new_cypher) / P(current_cypher)
                #If heads, replace the current cypher with the new one even if it's worse than the current cypher. This will
                #help the algorithm to break out of local minima
                uniform_value = np.random.uniform(0, 1)
                div = np.exp(new_cypher.plausibility - cypher.plausibility)
                if uniform_value < div:
                    cypher = new_cypher
            
            #Every 1000 iterations, print out the current best cypher
            if i % 1000 == 0:
                print(f"Iteration {i} - Plausibility: {self.best_cypher.plausibility}")
                print(self.best_cypher.decode(text))

        return self.best_cypher, self.best_cypher.decode(text)
    
    def calculate_plausibility(self, cypher_map:'Cypher') -> float:
        """Test the plausibility of the given map

        Args:
            map (dict): map to test

        Returns:
            float: plausibility of the map
        """
        #Calculate the plausibility of the cypher, starting as if the first character is a space
        plausability = 0
        text = " " + self.text

        for i, char in enumerate(text[:-1]):
            first_char      = cypher_map.decode(char)
            second_char     = cypher_map.decode(text[i+1])
            plausability   += self.transition_matrix.loc[first_char, second_char]
        
        return plausability