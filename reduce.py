import sys

current_word = None
current_time = 0
current_len = 0
word = None
# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    # parse the input we got from mapper.py
    line = line.strip()
    word, wordlen, time = line.split('\t')
    # convert word_len (currently a string) to double,time to a int
    try:
        wordlen = float(wordlen)
        time = int(time)
    except:
        # count was not a number, so silently
        # ignore/discard this line
        continue
    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_time += time
        current_len += wordlen
    elif current_word:
        # write result to STDOUT
        print('%s\t%s' % (current_word, current_len / float(current_time)))
    current_len = wordlen
    current_word = word
    current_time = time
# do not forget to output the last word if needed!
if current_word == word and current_word is not None:
    print('%s\t%s' % (current_word, current_len / float(current_time)))
