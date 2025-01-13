# Spam filter with Naive Bayes

This mini-project demonstrates a spam filter using the multinomial Naive Bayes algorithm. Using the [SMS Spam collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset , we implement the algorithm to classify the messages as spam or ham (not-spam). This project builds upon a project on this topic shared by [DataQuest](https://www.dataquest.io/blog/data-science-projects-for-beginners-with-source-code/#project13).


## Application of the Multinomial Naive Bayes algorithm
Essentially we calculate word frequencies, prior probabilities, and conditional probabilities to make predictions.
Each sms is broken down into its constituent words $(w_1, w_2, \dotsc , w_n)$.
For each new message, we need to calculate the following probabilities to classify as spam or ham.

$
P(Spam| w_1, w_2, \dotsc , w_n) \propto P (Spam)  \prod\limits_{i=1}^{n} P(w_i | Spam )
$

$
P(Ham| w_1, w_2, \dotsc , w_n) \propto P (Ham)  \prod\limits_{i=1}^{n} P(w_i | Ham )
$

In order to calculate the conditional probabilities $ P(w_i | Spam )$ and $P(w_i | Ham )$, we will use the following standard formulas:

$
P(w_i | Spam ) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}}
$

$
P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}}
$

In the above:

- $N_{w_i|Spam}$ = number of times the word $w_i$ appears in spam messages.
- $N_{w_i|Ham}$ = number of times the word $w_i$ appears in ham messages.
- $N_{Spam}$ = total number of words in the Spam messages.
- $N_{Ham}$ = total number of words in the Ham messages.
- $N_{Spam}$ = number of words in the total vocabulary.
- $\alpha$ = Laplace smoothing parameter (default is 1).

### Tools used
- Python programming language
- Jupyter notebook environment
- pandas 

The Jupyter notebook "Spam_Filter.ipynb" contains all the codes and the results. It is thoroughly commented and self-explanatory.