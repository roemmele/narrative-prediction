
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
    
    #  blog_link LIKE "%blogspot.com%" OR\
    # blog_link LIKE "http://facebook.com%" OR\
    #  blog_link LIKE "%livejournal.com%" OR\
    # OR blog_link LIKE "%opensalon.com%" (blog)
    # blog_link LIKE "%xanga.com%")\
    #blog_link LIKE "%blogsome.com%" OR\
    #blog_link LIKE "%tumblr.com%" OR\
    # blog_link LIKE "%scribd.com%" OR\ (*contains spam)
    # http://allfreelancewriting.com
    # http://www.redheadwriting.com (blog)
    #deviantart.com (blogs)
    #blog_link LIKE "%typepad.com%"  (blogs, but also news and promotional content)

    #Maybe:
    #http://maryrobinettekowal.com (blog?)
    #http://www.publetariat.com/
    #http://www.nanowrimo.org (blog about writing?)
    #http://journal.neilgaiman.com/ (blog about writing?)
    #http://www.authonomy.com (some writing, some commentary about writing)
    #http://www.publishersweekly.com/ (commentary)
    #http://fictionwritersreview.com (commentary)
    #http://www.shewrites.com (personal stories)
    #http://www.murdershewrites.com/ (personal stories)
    #http://www.aliettedebodard.com
    # http://www.smashwords.com (a lot of summaries?)
    # http://www.fanstory.com/ (a lot of poetry-style stories)
    #open.salon.com%"
    #http://www.benjaminrosenbaum.com/blog
    #http://www.lawrencemschoen.com
    #http://kentbrewster.com
    #"%kentbrewster.com%"
    # blog_link LIKE "%webfictionguide.com%" OR (forums)
    #OR blog_link LIKE "%fantasyhotlist.blogspot.com%"
    # blog_link LIKE "%published.com%" OR
     #OR blog_link LIKE "%fantasybookcritic.blogspot.com%"
    #  blog_link LIKE "%literotica.com%" OR
    #blog_link LIKE "%writerswrite.com%" OR
    #OR blog_link LIKE "%one-story.com%"
    #OR blog_link LIKE "%crossedgenres.com%"
    # blog_link LIKE "%storybird.com%"
    #  blog_link LIKE "%bookpage.com%" OR
    #OR blog_link LIKE "%abelard.org%"
    #blog_link LIKE "%jacketflap.com%" OR
    #http://romance-novels.alltop.com (a lot of blog-like stories interspersed)
    #blog_link LIKE "%manybooks.net%" OR (book summaries/short excerpts)

    #fiction (tons!):
    #blog_link LIKE "%wattpad.com%" OR
    #blog_link LIKE "%writerscafe.org%" OR
    
#     query = 'SELECT id\
#         FROM story\
#         WHERE (blog_link LIKE "%absolutewrite.com%" OR\
#                blog_link LIKE "%authonomy.com%" OR\
#                blog_link LIKE "%scribd.com%" OR\
#                blog_link LIKE "%bookpage.com%" OR\
#                blog_link LIKE "%fanfiction.net%" OR\
#                blog_link LIKE "%fanstory.com%" OR\
#                blog_link LIKE "%fictionwritersreview.com%" OR\
#                blog_link LIKE "%goodreads.com%" OR\
#                blog_link LIKE "%guidetoliteraryagents.com%" OR\
#                blog_link LIKE "%jacketflap.com%" OR\
#                blog_link LIKE "%publetariat.com%" OR\
#                blog_link LIKE "%published.com%" OR\
#                blog_link LIKE "%publishersweekly.com%" OR\
#                blog_link LIKE "%shewrites.com%" OR\
#                blog_link LIKE "%smashwords.com%" OR\
#                blog_link LIKE "%http://tor.com%" OR\
#                blog_link LIKE "%http://typepad.com%" OR\
#                blog_link LIKE "%urbis.com%" OR\
#                blog_link LIKE "%wattpad.com%" OR\
#                blog_link LIKE "%writerscafe.org%" OR\
#                blog_link LIKE "%writersdigest.com%" OR\
#                blog_link LIKE "%writersonlineworkshops.com%" OR\
#                blog_link LIKE "%writerswrite.com%" OR\
#                blog_link LIKE "%writing.com%" OR\
#             AND content_extract NOT LIKE "Start downloading documents right away.%"\
# ORDER BY id' #returns 13118750 stories in MySQL db

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
        query += ' WHERE NOT is_spam AND NOT is_duplicate'# AND NOT is_adult' #ignore stories that are null, duplicate, or spam
        
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

def make_seqs_table(story_ids, n_sent, db_filepath):
    #import pdb;pdb.set_trace()
    #create a new table to store sequences in index
    connection = sqlite3.connect(db_filepath)
    cursor = connection.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS sequence(                    id INTEGER PRIMARY KEY,                    story_id INTEGER,                    text TEXT)")
    
    n_seq = 0
    for story_id in story_ids:
        story = get_stories(story_id)
        seqs = split_into_seqs(story, n_sent)
        n_seq += len(seqs)
        cursor.executemany("INSERT INTO sequence(id, story_id, text)                            VALUES (?, ?, ?)", [(None, story_id, seq) for seq in seqs])

    connection.commit()
    connection.close()
    
    return n_seq


# In[6]:

# class StoryYielder():
#     #retrieves full stories one by one
#     def __init__(self, story_ids):
#         self.story_ids = story_ids
#     def __iter__(self):
#         #retrieve from stories table
#         for story_id in self.story_ids:
#             story = get_stories(story_id)
#             assert(story)
#             yield story
    

# class SequenceYielder():
#     #retrieves story segments one by one
#     def __init__(self, n_sent, story_ids, db_filepath):
#         self.n_sent = n_sent
#         self.story_ids = story_ids
#         self.db_filepath = db_filepath
#         self.n_seq = make_seqs_table(self.story_ids, n_sent, self.db_filepath)
#     def __iter__(self):
#         #retrieve from seqs table - sqlite3 ids start at 1
#         for seq_id in range(1, self.n_seq + 1):
#             #import pdb;pdb.set_trace()
#             #retrieve next - in sqlite ids start at 1
#             seq = get_seqs(seq_id, self.db_filepath)
#             assert(seq)
#             yield seq
        

# class SimilarityIndex():
#     def __init__(self, filepath, story_ids=None, stories=None, n_sent=None, min_freq=2):
#         '''either retreive stories from story ids in db, or index given stories'''
#         self.n_sent = n_sent
#         self.story_ids = story_ids
#         self.stories = stories
#         self.min_freq = min_freq
#         self.filepath = filepath
#         self.name = filepath.split("/")[-1]
#         if self.filepath[-1] != "/":
#             self.filepath += "/"
#         if os.path.isdir(self.filepath):
#             #try to load existing index if name given
#             self.load_index()
#         else:
#             assert(self.story_ids or self.stories)
#             #assert(self.stories is None if self.story_ids)
#             #assert(self.stories is None if self.n_sent)
#             #if index by this name doesn't exist, create new index
#             os.mkdir(self.filepath)
#             if not self.stories:
#                 #retrieve stories from db
#                 if self.n_sent:
#                     #import pdb;pdb.set_trace()
#                     #stories are sequences of self.n_sent sentences
#                     stories = SequenceYielder(self.n_sent, self.story_ids,
#                                                   db_filepath=self.filepath + self.name + ".seqs.db")
#                     #print "building index for", stories.n_seq, "story sequences of", self.n_sent, "sentences each"
#                 else:
#                     stories = StoryYielder(self.story_ids)
#             self.make_index(stories)
#             self.save_index()
    
#     def load_index(self):
#         print "loading index", self.name
#         if os.path.isdir(self.filepath + self.name + ".story_ids"):
#             with open(self.filepath + self.name + ".story_ids", 'rb') as f:
#                 self.story_ids = pickle.load(f)
#         self.lexicon = corpora.Dictionary.load(self.filepath + self.name + ".lexicon")
#         self.model = models.TfidfModel.load(self.filepath + self.name + ".model")
#         self.index = similarities.MatrixSimilarity.load(self.filepath + self.name + ".index")
                   
#     def save_index(self):
#         if self.story_ids:
#             with open(self.filepath + self.name + ".story_ids", 'wb') as f:
#                 pickle.dump(self.story_ids, f)
#         self.lexicon.save(self.filepath + self.name + ".lexicon")
#         self.model.save(self.filepath + self.name + ".model")
#         self.index.save(self.filepath + self.name + ".index")

#         print "saved index to", self.name, "folder"
                   
    
#     def make_index(self, seqs):
#         print "building index for", len([seq for seq in seqs]), "story sequences"
#         self.lexicon = corpora.Dictionary([tokenize(seq) for seq in seqs])
#         self.lexicon.filter_extremes(no_below=self.min_freq)
#         print "generated lexicon of", len(self.lexicon.keys()), "words with frequency >=", self.min_freq
#         self.lexicon.compactify()
#         #import pdb;pdb.set_trace()
#         corpus = [self.lexicon.doc2bow(tokenize(seq)) for seq in seqs]
#         self.model = models.TfidfModel(corpus, id2word=self.lexicon, normalize=True)
#         self.index = similarities.MatrixSimilarity(self.model[corpus])


#     def get_similar_seqs(self, seq, n_best=1):
#         #import pdb;pdb.set_trace()
#         seq = self.lexicon.doc2bow(segment_and_tokenize(seq))
#         scores = self.index[self.model[seq]]
#         best_ids = numpy.argsort(scores)[::-1]
#         if self.n_sent:
#             #use sequence ids for retrieval from seq db - sqlite ids start at 1
#             best_ids = list(best_ids[:n_best] + 1)
#             seqs = get_seqs(seq_ids=best_ids, db_filepath=self.filepath + self.name + ".seqs.db")
#         elif self.story_ids:
#             #use story ids for retrieval from story db
#             best_ids = [self.story_ids[id] for id in best_ids[:n_best]]
#             seqs = get_stories(story_ids=best_ids)
#         elif self.stories:
#             #stories already loaded in memory
#             seqs = [self.stories[id] for id in best_ids[:n_best]]     
#         return seqs
    


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



