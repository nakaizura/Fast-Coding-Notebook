import re
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import split_sentences
from gensim import corpora, models, similarities

# ASR文本去噪（可从数千字到几句话）
# 1特殊字符，标点数字字母
# 2文本去重，重复句和重复词
# 3短句删除
# 4textrank做抽取式摘要
# 5相似句去重

def clearn_str(string):
    # 筛除掉中文标点
    #string = re.sub(r'[＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。 ]', '', string)
    # 筛除掉英文标点
    string = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', '', string)
    return string
def clearstop(s):
    stopword=['oh','uh','huh','wow','la','yeah','mm','mmm','mmmm','hey','hello','hi','subscribe','channel','video','thank']
    #stopwords1 = stopwords.words('english')
    #print(stopwords1)
    s=s.replace('!','.')
    sen=s.split('.')
    #print(len(sen))
    res=[]
    for i in sen:
        if len(i)>4:
            stop=0
            for j in i.split():
                if j in stopword:
                    stop=1
                    break
            if stop==0: res.append(i)
    #print(len(res))
    return '.'.join(res)


#summary
def f(seq):  # Order preserving unique sentences - sometimes duplicate sentences appear in summaries
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]
def summary(x, perc):  # x input document, perc: percentage of the original document to keep
    if len(split_sentences(x)) > 10:
        test_summary = summarize(x, ratio=perc, split=True)
        test_summary = '\n'.join(map(str, f(test_summary)))
    else:
        test_summary = x
    return test_summary


def get_tfidf(words_lists):
    texts = words_lists
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id)

    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    return tfidf, dictionary, corpus, feature_cnt
def get_semantic_similarity_for_line(words_list1, tfidf, dictionary, corpus, feature_cnt):
    kw_vector = dictionary.doc2bow(words_list1)  # (jieba.lcut(keyword))
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[kw_vector]]
    return sim


def topic(text):
    # stopword
    stopwords1 = stopwords.words('english')
    stopwords1.extend(['little','bit','going','good','got','three'])

    stopwords2=[]
    with open('./data/stopword.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            lline = line.strip()
            stopwords2.append(lline)

    sentences = []
    segs = [clearn_str(i).split() for i in text.split('.')]
    segs = [list(filter(lambda x: x not in stopwords2, i)) for i in segs]
    sentences.append(segs)
    sentences=sentences[0]
    dictionary = corpora.Dictionary(sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)
    #topic_list=lda.print_topic(0, topn=3)
    #topic
    topic_list=lda.get_document_topics(corpus)
    return topic_list[0]


s="oh man,  paralyzed.  absolutely san seasonal is just uh huh. what's going on everybody here with emmy maiden japan i'm super excited today we're going to be doing a collaboration video we're going to make an american classic lt is basically my favorite sandwich i'm very passionate about it i make my bio tea i guess i don't know very specific to me it's really easy that's kind of. what's one of these log and saw well it looks like bl t heaven but what does it like. very simple way of making b t so i'm very excited to make it she's going to be doing some home fries so it can be awesome very fresh air's fresh. roddy i god's well thanks to watson remember thumb this video. and we'll do local collaboration video and hope you guys enjoy i'll be linking her so you can click on your face you want to go subscribe check her out i'll be also winking at the end of the video and probably in the description box so please check out. sorry moorish,  you really do have to ask yourself right now,  and if you do,  please let me know how yours turns out. so it's getting the kitchen and make an american classic guilty. cool down if you thought i beg was a gangster wrapper.  wow wow wow wow wow wow. i want to show any sort of assembled i don't like a sandwich ich my biggest thing is having. uh la la la! everything equally distributed because i hate the sandwich button and everything falls out sometimes it's just bound to happen because of the mayonnaise or mustard the economics tend to make it a little slippery huge is kinda funny but thinly sized tomatoes just so you know it's like three slices three slices two pieces of lettuce six pieces bacon harmony pieces. thanks so what's and i'll see you next sonia hungrily kitch kitchen yeah wax jump with pay pale absolutely amazing you really do have to give this. just evenly lined up,  so let me just give you,  yeah,  maybe a little.  especially cut tomatoes. don't hold me so we'd bike can you actually hey ain't it'll mb bike bike you actually wanted that gold golden colo. sure,  knives are really super recording cut off. mm. i mean that's a pretty yeah yeah. the science house needs. that's not crazy but thin enough to where you can stack these huge chunky. buyers. crazy tomatoes cut that out i'm eager yeah definitely definitely probably maybe maybe one safety safety and i'll wash the lettuce log xj. mm. hey! if we. mm. mm. mm. hey! once you have all your ingredients basically going to do is get a little assembly line going with your cooked bacon lettuce tomato and toasted bread with livid a mayo once you're done with all that ought to do a stack at all together and make a beautiful. mmmm. mm. log. hi! anthony. yeah. mm. gosh j. if we did. sure,  knives are really super recording cut off. so there you guys have it a beautiful balanced t that in every bite yet the perfect amount of bacon lettuce tomato please go check out emmy's channel off for the home fries recipe and i'll see you guys next time. mm. hmm. just evenly lined up,  so let me just give you,  yeah,  maybe a little.  especially cut tomatoes. am so ready to dig into this.  looks beautiful right?  so this is.  this is awesome cookies though. hey! make sure,  yeah. if we. it's governing. mm. mm. this is a perfect! mm. perfect.  perfect lunch.  i love that the facts.  big mcs state cut fries really tender. i'm poised doing his breathing. we do! mm. i had so much fun this was such a cool collaboration i'm so excited ever chance to do this love the fries. log. think we devoured most of it i think it looks great hopefully enjoyed my guilt and my guys enjoy this video please go subscribe to enemy's channel she's so awesome she has admitted she takes food like a mad person she runs smooth and i love people who love foods of his physical checkout she's an amazing channel. mmmm. and absolutely fighting entity somehow run somewhere around here so please go check it out also thank you so much pertama this is awesome awesome set we get the filament yeah just super excited i've never actually had a chance to film like a profession. hi! thank you so much to taste!  if you guys enjoy this video,  please subscribe,  thumbs up and sea you guys in my next video!oh man,  paralyzed.  absolutely san seasonal is just uh huh. what's going on everybody here with emmy maiden japan i'm super excited today we're going to be doing a collaboration video we're going to make an american classic lt is basically my favorite sandwich i'm very passionate about it i make my bio tea i guess i don't know very specific to me it's really easy that's kind of. what's one of these log and saw well it looks like bl t heaven but what does it like. very simple way of making b t so i'm very excited to make it she's going to be doing some home fries so it can be awesome very fresh air's fresh. roddy i god's well thanks to watson remember thumb this video. and we'll do local collaboration video and hope you guys enjoy i'll be linking her so you can click on your face you want to go subscribe check her out i'll be also winking at the end of the video and probably in the description box so please check out. sorry moorish,  you really do have to ask yourself right now,  and if you do,  please let me know how yours turns out. so it's getting the kitchen and make an american classic guilty. cool down if you thought i beg was a gangster wrapper.  wow wow wow wow wow wow. i want to show any sort of assembled i don't like a sandwich ich my biggest thing is having. uh la la la! everything equally distributed because i hate the sandwich button and everything falls out sometimes it's just bound to happen because of the mayonnaise or mustard the economics tend to make it a little slippery huge is kinda funny but thinly sized tomatoes just so you know it's like three slices three slices two pieces of lettuce six pieces bacon harmony pieces. thanks so what's and i'll see you next sonia hungrily kitch kitchen yeah wax jump with pay pale absolutely amazing you really do have to give this. just evenly lined up,  so let me just give you,  yeah,  maybe a little.  especially cut tomatoes. don't hold me so we'd bike can you actually hey ain't it'll mb bike bike you actually wanted that gold golden colo. sure,  knives are really super recording cut off. mm. i mean that's a pretty yeah yeah. the science house needs. that's not crazy but thin enough to where you can stack these huge chunky. buyers. crazy tomatoes cut that out i'm eager yeah definitely definitely probably maybe maybe one safety safety and i'll wash the lettuce log xj. mm. hey! if we. mm. mm. mm. hey! once you have all your ingredients basically going to do is get a little assembly line going with your cooked bacon lettuce tomato and toasted bread with livid a mayo once you're done with all that ought to do a stack at all together and make a beautiful. mmmm. mm. log. hi! anthony. yeah. mm. gosh j. if we did. sure,  knives are really super recording cut off. so there you guys have it a beautiful balanced t that in every bite yet the perfect amount of bacon lettuce tomato please go check out emmy's channel off for the home fries recipe and i'll see you guys next time. mm. hmm. just evenly lined up,  so let me just give you,  yeah,  maybe a little.  especially cut tomatoes. am so ready to dig into this.  looks beautiful right?  so this is.  this is awesome cookies though. hey! make sure,  yeah. if we. it's governing. mm. mm. this is a perfect! mm. perfect.  perfect lunch.  i love that the facts.  big mcs state cut fries really tender. i'm poised doing his breathing. we do! mm. i had so much fun this was such a cool collaboration i'm so excited ever chance to do this love the fries. log. think we devoured most of it i think it looks great hopefully enjoyed my guilt and my guys enjoy this video please go subscribe to enemy's channel she's so awesome she has admitted she takes food like a mad person she runs smooth and i love people who love foods of his physical checkout she's an amazing channel. mmmm. and absolutely fighting entity somehow run somewhere around here so please go check it out also thank you so much pertama this is awesome awesome set we get the filament yeah just super excited i've never actually had a chance to film like a profession. hi! thank you so much to taste!  if you guys enjoy this video,  please subscribe,  thumbs up and sea you guys in my next video!what's going on everybody here with emmy maiden japan i'm super excited today we're going to be doing a collaboration video we're going to make an american classic lt is basically my favorite sandwich i'm very passionate about it i make my bio tea i guess i don't know very specific to me it's really easy that's kind of. very simple way of making b t so i'm very excited to make it she's going to be doing some home fries so it can be awesome very fresh air's fresh. and we'll do local collaboration video and hope you guys enjoy i'll be linking her so you can click on your face you want to go subscribe check her out i'll be also winking at the end of the video and probably in the description box so please check out. so it's getting the kitchen and make an american classic guilty. i want to show any sort of assembled i don't like a sandwich ich my biggest thing is having. everything equally distributed because i hate the sandwich button and everything falls out sometimes it's just bound to happen because of the mayonnaise or mustard the economics tend to make it a little slippery huge is kinda funny but thinly sized tomatoes just so you know it's like three slices three slices two pieces of lettuce six pieces bacon harmony pieces. just evenly lined up,  so let me just give you,  yeah,  maybe a little.  especially cut tomatoes. sure,  knives are really super recording cut off. i mean that's a pretty yeah yeah. that's not crazy but thin enough to where you can stack these huge chunky. crazy tomatoes cut that out i'm eager yeah definitely definitely probably maybe maybe one safety safety and i'll wash the lettuce log xj. mm. hey! if we. mm. mm. mm. once you have all your ingredients basically going to do is get a little assembly line going with your cooked bacon lettuce tomato and toasted bread with livid a mayo once you're done with all that ought to do a stack at all together and make a beautiful. mmmm. mm. log. hi! anthony. yeah. mm. gosh j. if we did. so there you guys have it a beautiful balanced t that in every bite yet the perfect amount of bacon lettuce tomato please go check out emmy's channel off for the home fries recipe and i'll see you guys next time. mm. hmm. am so ready to dig into this.  looks beautiful right?  so this is.  this is awesome cookies though. make sure,  yeah. it's governing. mm. this is a perfect! perfect.  perfect lunch.  i love that the facts.  big mcs state cut fries really tender. i'm poised doing his breathing. we do! i had so much fun this was such a cool collaboration i'm so excited ever chance to do this love the fries. think we devoured most of it i think it looks great hopefully enjoyed my guilt and my guys enjoy this video please go subscribe to enemy's channel she's so awesome she has admitted she takes food like a mad person she runs smooth and i love people who love foods of his physical checkout she's an amazing channel. and absolutely fighting entity somehow run somewhere around here so please go check it out also thank you so much pertama this is awesome awesome set we get the filament yeah just super excited i've never actually had a chance to film like a profession. thank you so much to taste!  if you guys enjoy this video,  please subscribe,  thumbs up and sea you guys in my next video!"
#s="it was wonderful bunch of slabs of baby backs on uh meat looks like he's getting ready for the for the indoor grill oh he's got his homemade barbecue sauce we put that on the website they'd go that's wonderful. this is nick from next guitar videos and you're watching her's racks. thank! mm. lo cooking bacon. so we sold them i'll run it's awesome killing veterinarian harry lawrence and columbia columbia south carolina hometown boy. morgan gan can smell the bacon welcome to eric meal time today i'm going to be building a sandwich. is there really any leftover bacon i've never really had left over it doesn't exist. and i try to do this in real time so let's get started here we go. begging if you're going to do. some nice bread.  bit of mayonnaise on the bread. syrup is good,  jellies are good,  is always such a good heavy sugar content and then use some stuff called pot wood which is a kind of a pope.  probably jl lyne hot sauce. i'd like to put a little bit of seasoning salt and pepper ride on the bottom there. oh,  that was good.  and this is our fresh lettuce from the garden where autonomy. next move on this is ab lt so the main ingredient right here it was an epic meal time bacon weave it's a double sixteen pieces bacon she's gonna dump. on top there! this hypothesis good i mean i could tell you there's our vacant netflix yumi i think we should because a b lp what this is like a pig is a pig candy beale a big candy specialty so we've got our parade and i say that since we are here let's tissue all. that took a bit of preparation. right? all right so that's that we're going to talk a little bit of ham. ha! just a little bit of ham left over hair practices case we've got the bank got the ham we're going to go for some tomatoes. manny his own we've got to cut half what's cutting quarters share with everybody else somebody out here. mass of tomatoes share a little bit ron,  a little bit more pepper. equality. seasoning,  salt,  avocados,  scoop alcoholism on hair. plunge. hi! it's the finishing touch and we use a nice dukes course give a favour i used dick's wire hang fixtures aid increment. express express express. dish dish. making hair triad area meal times we've got that we're going to top. hi! with cheese helps cheese. wow. cheese slices on top. it can be built. eager stick together a little bit sticking together that's not good. by local funk,  global spaces,  sustainable and every change you begin thug gema. sticking together. all right. gosh. we'll come out just as well. mm. uh,  she got to cheese slices on their. lo cooking bacon. you've got to put some lattice on top of that. ok. well,  not much. there. and i try to do this in real time so let's get started here we go. and a little bit more. on top there! manny on top of that. seasoning salt. pepper. and on top. plunge. bread sandwich right there. mm. seasoning,  salt,  avocados,  scoop alcoholism on hair. alright let's eat this thing. mass of tomatoes share a little bit ron,  a little bit more pepper. lt.  eric meal time. oh. with cheese helps cheese. oh it's not bad. mm. sticking together. gunboat mm. just a little bit of ham left over hair practices case we've got the bank got the ham we're going to go for some tomatoes. but some governments,  ha. there. three. we'll come out just as well. i forgot what was the purpose of the video again? round four. mm. mm. well,  not much. round five. but. uh uh uh uh uh uh uh uh uh uh. and a little bit more. this video was a bad idea. what was i thinking? thanks! yep. yep. hmm. mm. so we go at meal time. watch out! mm. bread sandwich right there. i don't think i'll do another one of these.this is nick from next guitar videos and you're watching her's racks. thank! mm. lo cooking bacon. morgan gan can smell the bacon welcome to eric meal time today i'm going to be building a sandwich. and i try to do this in real time so let's get started here we go. some nice bread.  bit of mayonnaise on the bread. i'd like to put a little bit of seasoning salt and pepper ride on the bottom there. next move on this is ab lt so the main ingredient right here it was an epic meal time bacon weave it's a double sixteen pieces bacon she's gonna dump. on top there! that took a bit of preparation. all right so that's that we're going to talk a little bit of ham. just a little bit of ham left over hair practices case we've got the bank got the ham we're going to go for some tomatoes. mass of tomatoes share a little bit ron,  a little bit more pepper. seasoning,  salt,  avocados,  scoop alcoholism on hair. plunge. hi! express express express. making hair triad area meal times we've got that we're going to top. with cheese helps cheese. cheese slices on top. eager stick together a little bit sticking together that's not good. sticking together. all right. we'll come out just as well. uh,  she got to cheese slices on their. you've got to put some lattice on top of that. ok. well,  not much. there. and a little bit more. manny on top of that. seasoning salt. pepper. and on top. bread sandwich right there. mm. alright let's eat this thing. lt.  eric meal time. oh. oh it's not bad. mm. gunboat mm. but some governments,  ha. there. three. i forgot what was the purpose of the video again? round four. mm. mm. round five. but. uh uh uh uh uh uh uh uh uh uh. this video was a bad idea. what was i thinking? thanks! yep. yep. hmm. so we go at meal time. watch out! mm. i don't think i'll do another one of these."
#

def clearasr(s):
    s=clearstop(s)
    mysummary = summary(s, 0.15)
    #print(mysummary,len(mysummary.split('.')))
    summa=[i.split() for i in mysummary.split('.')]
    #print(len(summa))
    tfidf, dictionary, corpus, feature_cnt=get_tfidf(summa)
    score=[get_semantic_similarity_for_line(i, tfidf, dictionary, corpus, feature_cnt) for i in summa]
    hsim=[]
    for i in range(len(summa)):
        for j in range(i+1,len(summa)):
            if score[i][j]>0.8:
                hsim.append(j)
    #print(hsim)
    res=[]
    for i in range(len(summa)):
        if i not in hsim:
            res.append(' '.join(summa[i]))
    #print('.'.join(res))
    return '.'.join(res)

print(clearasr(s))


