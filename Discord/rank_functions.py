import re, sqlite3

def parse_command(message):
    try:
        #let's try to extract the message
        content = message.content
    except:
        #if it fails for whatever reason, return emptyhanded
        return
    
    #This will grab the message. Words or discord IDs
    regex_string = r"((?!\s)<@\d+>)|(\w+)"

    #Split message 
    try:
        raw_split = re.findall(regex_string, content, flags=re.I)
        split = []
        #split will be a list of tuples, so we want to reduce it to a list with only the non-blank values
        for i in raw_split:
            if i[0] == '':
                split.append(i[1])
            else:
                split.append(i[0])
        
        print(split)

    except:
        return 'an error occurred with your input'

    #enumerate the functions with respective keys
    function_list = {
        'record': record_match,
        'stats': display_stats,
        'dispute': dispute_match,
        'predict': predict_match
    }

    #if the first 'word' after !rankbot is a valid function, execute the function by passing through a list with the remaining values of the array
    if split[1] in function_list.keys():
        try:
            msg = function_list[split[1]](split[2:])
        except:
            return 'Exception in split'
    else:
        #if it's invalid, return nothing
        return
    
    return msg
        

def record_match(content_li):
    firstID , vs, secondID = content_li[0:3]
    valid_vs = ['vs', 'v', 'versus']

    if is_valid_id(firstID) and (vs in valid_vs) and is_valid_id(secondID):
        print(firstID, secondID)
    else:
        return 'Invalid format! Try again'



def display_stats(content_li):
    None

def dispute_match(content_li):
    None

def predict_match(content_li):
    None

def is_valid_id(candidate):
    
    regex_string = r"<@\d+>"
    match = re.search(regex_string, candidate)
    if match:
        return True
    else:
        return False
