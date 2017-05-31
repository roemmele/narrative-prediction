
# coding: utf-8

# In[1]:

import sqlite3, os, pickle, MySQLdb
from gensim import corpora, models, similarities
from gensim.matutils import Dense2Corpus, dense2vec
from models.transformer import *


# In[5]:

#sqlite db info
parsed_db_filepath = "../stories.db"
if os.path.isdir("/Volumes/G-DRIVE mobile with Thunderbolt/"):
    parsed_db_filepath = "/Volumes/G-DRIVE mobile with Thunderbolt/stories.db"

def get_story_ids(db_type='mysql', offset=0, n_stories=None, min_length=None, max_length=None, sites=[], filter=False):

    assert(db_type in ('mysql', 'sqlite'))

    query = 'SELECT id FROM story'

    if db_type == 'mysql':
        db = MySQLdb.connect(host='saysomething.ict.usc.edu', user="all_data_user",
                             passwd="all_data_pw", db_name="all_stories")
        
        if sites: #specified list of websites to retrieve from
            where_clause = ''
            for idx, site in enumerate(sites):
                where_clause += ' blog_link LIKE "%' + site + '%"'
                if idx < len(sites) - 1:
                     where_clause += ' OR'
        
        if min_length:
            query += " AND LENGTH(content_extract) >= " + str(min_length)
        
        if max_length:
            query += " AND LENGTH(content_extract) <= " + str(max_length)
        
        query += where_clause
                        
    elif db_type == 'sqlite':
        db = sqlite3.connect(parsed_db_filepath)
        query += ' WHERE NOT is_spam AND NOT is_duplicate AND NOT is_adult' #ignore stories that are null, duplicate, or spam
        
        if min_length:
            query += " AND LENGTH(text) >= " + str(min_length)
        
        if max_length:
            query += " AND LENGTH(text) <= " + str(max_length)

    if n_stories:
        #limit number of stories returned
        query += " LIMIT " + str(n_stories)
    if offset > 0:
        query += " OFFSET " + str(offset)
        
    cursor = db.cursor()
    cursor.execute(query)
    story_ids = [int(id[0]) for id in cursor.fetchall()]
    
    cursor.close()
    db.close()
    
    return story_ids

def get_stories(story_ids):
    if type(story_ids) is not list:
        #only one story id given
        story_ids = [story_ids]
    db = sqlite3.connect(parsed_db_filepath)
    cursor = db.cursor()
    stories = []
    for story_id in story_ids:
        cursor.execute("SELECT text FROM story WHERE id = ?", (story_id,))
        story = cursor.fetchone()
        if story:
            story = unicode(story[0])
        else:
            story = ""
        stories.append(story)
    cursor.close()
    db.close()
    if len(stories) == 1:
        #if only one story requested, return single string
        stories = stories[0]
    return stories

def filter_nonstory_ids(story_ids, min_length=None, max_length=None):
    db = sqlite3.connect(parsed_db_filepath)
    cursor = db.cursor()
    true_story_ids = []
    query = "SELECT id FROM story WHERE id = ?            AND NOT is_spam AND NOT is_duplicate AND NOT is_adult"
    if min_length:
        query += " AND LENGTH(text) >= " + str(min_length)
    if max_length:
        query += " AND LENGTH(text) <= " + str(max_length)
    for story_id in story_ids:
        cursor.execute(query, (story_id,))
        if cursor.fetchone():
            true_story_ids.append(story_id) #story is valid
    cursor.close()
    db.close()
    return true_story_ids

def sort_ids_by_len(story_ids):
    chunk_size = 1000000
    story_lens = []
    #count number of words in each story and return sort order of ids
    for idx in range(0, len(story_ids), chunk_size):
        #lens = [len(segment_and_tokenize(story.strip())) for story in get_stories(story_ids[idx:idx+chunk_size])]
        #lengths are approximate in number of words because just splitting on
        #spaces rather than tokenizing (which takes too long)
        lens = [len(story.strip().split(" ")) for story in get_stories(story_ids[idx:idx+chunk_size])]
        story_lens.extend(lens)
        print "sorted", idx+chunk_size, "stories by length"
    story_ids = numpy.array(story_ids)[numpy.argsort(story_lens)]
    return list(story_ids)

def split_into_seqs(story, n_sent=5, max_length=500):
    #split story into sequences of n_sent sentences
    #import pdb;pdb.set_trace()
    sentences = segment(story)
    seqs = []
    for seq_start in range(0, len(sentences), n_sent):
        seq_end = seq_start + n_sent
        if seq_start == len(sentences) - 1:
            #only one sentence left, break
            #import pdb;pdb.set_trace()
            break
        elif len(sentences) - seq_start <= n_sent + 1:
            #less than n_sent sentences left in this story, so add null sentences to end
            #import pdb;pdb.set_trace()
            sentences.extend([""] * (n_sent - (len(sentences) - seq_start) + 1))
        sent_lengths = [len(sentence) for sentence in sentences[seq_start:seq_end]]
        if sum(sent_lengths) > max_length:
            #filter sequences longer than max_length number of characters
            continue
        seqs.append(" ".join(sentences[seq_start:seq_end]).strip())
    return seqs   
    

def get_input_output_seqs(stories, n_input_sent=4, max_length=500, mode='adjacent'):
    #segment stories into sequences of five sentences
    if type(stories) is not list:
        stories = [stories]
    input_seqs = []
    output_seqs = []
    for story in stories:
        #filter non-ascii characters
        story = story.encode('ascii', errors='ignore')
        seqs = split_into_seqs(story)
        for seq in seqs:
            seq = segment(seq)
            input_seq = seq[:-1]
            output_seq = seq[1:]
            if mode == "concat":
                input_seq = " ".join(input_seq)
                #output in concat mode is just last sentence in the segment
                output_seq = output_seq[-1]
            assert(len(input_seq) == n_input_sent)
            assert(len(output_seq) == n_input_sent)
            input_seqs.append(input_seq)
            output_seqs.append(output_seq)
    assert(len(input_seqs) == len(output_seqs))
    return input_seqs, output_seqs

def get_seqs(seq_ids, db_filepath):
    #import pdb;pdb.set_trace()
    if type(seq_ids) is not list:
        #only one seq id given
        seq_ids = [seq_ids]
    db = sqlite3.connect(db_filepath)
    cursor = db.cursor()
    cursor.execute("SELECT text FROM sequence WHERE id IN (" + ",".join(map(str, seq_ids)) + ")                    ORDER BY id")
    seqs = [text[0] for text in cursor.fetchall()]
    cursor.close()
    db.close()
    if len(seqs) == 1:
        #if only one sequence requested, return single string
        seqs = seqs[0]
    return seqs

def get_next_seq(seq_id, db_filepath):
    '''get the sequence that appears after the given sequence id in its respective story;
    if there's no sequence (i.e. the sequence id is last sequence of the story), return None'''

    db = sqlite3.connect(db_filepath)
    cursor = db.cursor()
    #first get story id for this sequence
    cursor.execute("SELECT story_id FROM sequence WHERE id = " + str(seq_id))
    story_id = cursor.fetchone()
    if not story_id:
        print "error: sequence ID", seq_id, "not found in", db_filepath
        return None
    else:
        story_id = story_id[0]
    #then get sequence that follows sequence ID, ensuring that the story ID is the same
    cursor.execute("SELECT text FROM sequence WHERE id = " + str(seq_id + 1) + " AND story_id = " + str(story_id))
    seq = cursor.fetchone()
    cursor.close()
    db.close()
    if not seq:
        return None
    else:
        return seq[0]

def get_n_seqs(db_filepath):
    #import pdb;pdb.set_trace()
    db = sqlite3.connect(db_filepath)
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM sequence")
    n_seqs = cursor.fetchone()[0]
    cursor.close()
    db.close()
    return n_seqs


def make_seqs_table(story_ids, n_sents, db_filepath):
    #import pdb;pdb.set_trace()
    #create a new table to store sequences in index
    db = sqlite3.connect(db_filepath)
    cursor = db.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS sequence(                    id INTEGER PRIMARY KEY,                    story_id INTEGER,                    text TEXT)")
    
    n_seqs = 0
    for story_idx, story_id in enumerate(story_ids):
        story = get_stories(story_id)
        seqs = split_into_seqs(story, n_sents)
        n_seqs += len(seqs)
        cursor.executemany("INSERT INTO sequence(id, story_id, text)                            VALUES (?, ?, ?)", [(None, story_id, seq) for seq in seqs])
        if story_idx % 50000 == 0:
            print "added", n_seqs, "sequences to", db_filepath, "for", story_idx, "stories..."

    db.commit()
    db.close()
    
    return n_seqs
    


# In[7]:

if __name__ == "__main__":
    import pdb;pdb.set_trace()
#     story_ids = get_trusted_story_ids(n_stories=100000)
    with open('../fiction_ids.txt', 'r') as id_file:
        story_ids = [int(story_id) for story_id in id_file.readlines()]
    stories = get_stories(story_ids[:50000])
#     sim_index = SimilarityIndex(filepath="stories_100000", story_ids=story_ids, n_sent=5, min_freq=10)
#     print "most similar stories:\n"
#     for story_id in sim_story_ids:
#         print stories[story_id]


# In[337]:

if __name__ == "__main__":
    seqs = split_into_seqs(get_stories(story_ids[50]))
    print seqs[0]
    import pdb;pdb.set_trace()
    sim_seqs = sim_index.get_similar_seqs(seqs[0], n_best=3)
    print sim_seqs[0]
    print sim_seqs[1]
    print sim_seqs[2]


# In[13]:

'''code to get site names of stories listed on novelsonline.info'''

if __name__ == "__main__":
    from bs4 import BeautifulSoup
    from urlparse import urlparse
    import pdb;pdb.set_trace()
    page_file = '../view-source_novelsonline.info.html'
    with open(page_file, 'r') as f:
        page = BeautifulSoup(f.read())
        sites = set([urlparse(link['href']).netloc for link in page.find_all('a')])
        for site in sites:
            print "blog_link LIKE \"%" + site + "%\" OR"


# In[16]:

'''code to get site names of stories listed on livejournal'''

if __name__ == "__main__":
    from bs4 import BeautifulSoup
    from urlparse import urlparse
    import pdb;pdb.set_trace()
    page_file = '../view-source_https___www.tumblr.com_search_harry+styles+fanfiction.htm'
    with open(page_file, 'r') as f:
        page = BeautifulSoup(f.read())
        sites = set([urlparse(link['href']).netloc for link in page.find_all('a')])
        for site in sites:
            print "blog_link LIKE \"%" + site + "%\" OR"


# In[ ]:



