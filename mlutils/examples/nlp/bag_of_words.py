import spacy
from collections import Counter
import pandas as pd
from mlutils.utils.textutils import BagOfWords, SpaCyTokenizerWrapper, SpaCySentenceBuilder, compute_word_count_vector

if __name__ == '__main__':
    text = ("Algorithmic bias has been cited in cases ranging from "
            "election outcomes to the spread of online hate speech.")

    nlp = spacy.load("en_core_web_sm")
    tokenizer = SpaCyTokenizerWrapper(nlp=nlp)
    bow = BagOfWords()

    bow.add_text(text=text, tokenizer=tokenizer)
    print(bow.bow)

    text1 = ("Algorithmic bias describes systematic and repeatable errors in a computer system "
             'that create unfair outcomes, such as privileging one arbitrary group of users over others. '
             'Bias can emerge due to many factors, including but not limited to the design of the algorithm '
             'or the unintended or unanticipated use or decisions relating to the way data is coded, collected, '
             'selected or used to train the algorithm. '
             'Algorithmic bias is found across platforms, including but not limited '
             'to search engine results and social media platforms, and can have impacts '
             "ranging from inadvertent privacy violations to reinforcing social biases of race, gender, sexuality, and ethnicity.")

    text2 = ('The study of algorithmic bias is most '
              'concerned with algorithms that reflect "systematic and unfair" discrimination. '
              'This bias has only recently been addressed in legal frameworks, such as the 2018 European Unions '
              'General Data Protection Regulation. More comprehensive regulation is needed as emerging technologies become increasingly advanced and opaque.')

    text3 = ('As algorithms expand their ability to organize society, '
              'politics, institutions, and behavior, sociologists have become '
              'concerned with the ways in which unanticipated output and manipulation '
              'of data can impact the physical world. Because algorithms are often considered to '
              'be neutral and unbiased, they can inaccurately project greater authority than human expertise, '
              'and in some cases, reliance on algorithms can displace human responsibility for their outcomes. '
              'Bias can enter into algorithmic systems as a result of pre-existing cultural, social, '
              'or institutional expectations; because of technical limitations of their design; or by being used '
              'in unanticipated contexts or by audiences who are not considered in the software  initial design.')

    text4 = ('Algorithmic bias has been cited in cases ranging from election outcomes to '
              'the spread of online hate speech. It has also arisen in criminal justice, healthcare, and hiring, '
              'compounding existing racial, economic, and gender biases. The relative inability of facial recognition '
              'technology to accurately identify darker-skinned faces has been linked to '
              'multiple wrongful arrests of men of color, an issue stemming from imbalanced datasets. '
              'Problems in understanding, researching, and discovering algorithmic bias persist due to the proprietary nature of algorithms, '
              'which are typically treated as trade secrets. Even when full transparency is provided, '
              'the complexity of certain algorithms poses a barrier to understanding their functioning. '
              'Furthermore, algorithms may change, or respond to input or output in ways that cannot be anticipated or '
              'easily reproduced for analysis. In many cases, even within a single website or application, there is no '
              'single "algorithm" to examine, but a network of many interrelated programs and data inputs, even between users of the same service.')

    docs = [text1, text2, text3, text4]

    # clear the bag of words
    bow.clear()

    doc_tokens = []
    for doc in docs:
        doc_tokens.append([tok.text.lower() for tok in nlp(doc)])

    print(f"Document tokens for document 0: {len(doc_tokens[0])}")

    # concatenate all lists together
    all_doc_tokens = []
    for tokens in doc_tokens:
        all_doc_tokens.extend(tokens)

    print(f"Number of total tokens in all docs: {len(all_doc_tokens)}")

    # vocabulary
    vocab = set(all_doc_tokens)
    vocab = sorted(vocab)

    print(f"Length of vocabulary: {len(vocab)}")
    count_vectors = []
    for tokens in doc_tokens:
        count_vectors.append(Counter(tokens))

    tf = pd.DataFrame(count_vectors)
    tf = tf.T.sort_index().T
    tf = tf.fillna(0).astype(int)
    print(tf)

    # we can do this with word-count vector
    wc_vec = compute_word_count_vector(corpus=docs)

    print(wc_vec)

    # bow.add_text(text=text2, tokenizer=tokenizer)
    # print(bow.most_common(n=5))
    #
    # # use sentence builder also
    # sentences = SpaCySentenceBuilder(nlp=nlp, expand=True)(text2)
    # print(sentences)
    #
    # # for every sentence we will get the tokens.
    # # This way we create a lexicon or vocabulary. i.e. a list of all the unique
    # # tokens in your corpus.
    # # Just like a dictionary of words at the library, a lexicon doesn’t contain any duplicates
    # bow = BagOfWords()
    #
    # for sentence in sentences:
    #     bow.add_text(text=sentence, tokenizer=tokenizer)
    #
    # print(f"Size of BoW {len(bow)}")
    # print(bow.bow)
    #
    # # we can create a Pandas DataFrame
    # df = bow.as_pandas_df(fill_na=True)
    #
    # print(df.head(n=15))




