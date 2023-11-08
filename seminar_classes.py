import math


class CountVectorizer:
    def __init__(self):
        self.list_of_unique_words = []

    def fit_transform(self, text):
        whole_text = ' '.join(text)
        united_list_of_words = whole_text.split()
        count_array = []
        for word in united_list_of_words:
            word_cleared = word.strip('?!.,:;-()=[]{}$').lower()
            if word_cleared not in self.list_of_unique_words:
                self.list_of_unique_words.append(word_cleared)
        for sentence in text:
            count_list = []
            for word in self.list_of_unique_words:
                count_list.append(sentence.lower().count(word))
            count_array.append(count_list)
        return count_array

    def get_feature_names(self):
        return self.list_of_unique_words


def tf_transform(count_matrix):
    tf_transform_list = []
    for row in count_matrix:
        tf_transform_row_list = []
        for word in row:
            tf_transform_row_list.append(round(word / sum(row), 3))
        tf_transform_list.append(tf_transform_row_list)
    return tf_transform_list


def idf_transform(count_matrix):
    idfs = []
    total_rows = len(count_matrix)
    total_words = len(count_matrix[0])
    for i in range(total_words):
        rows_i = 0
        for row in count_matrix:
            if row[i] > 0:
                rows_i += 1
        idfs.append(round(math.log((total_rows + 1) / (rows_i + 1)) + 1, 3))
    return idfs


class TfidfTransformer:
    @staticmethod
    def tf_transform(count_matrix):
        tf_transform_list = []
        for row in count_matrix:
            tf_transform_row_list = []
            for word in row:
                tf_transform_row_list.append(round(word / sum(row), 3))
            tf_transform_list.append(tf_transform_row_list)
        return tf_transform_list

    @staticmethod
    def idf_transform(count_matrix):
        idfs = []
        total_rows = len(count_matrix)
        total_words = len(count_matrix[0])
        for i in range(total_words):
            rows_i = 0
            for row in count_matrix:
                if row[i] > 0:
                    rows_i += 1
            idfs.append(round(math.log((total_rows + 1) / (rows_i + 1)) + 1, 3))
        return idfs

    def fit_transform(self, count_matrix):
        tfs = self.tf_transform(count_matrix)
        idfs = self.idf_transform(count_matrix)
        tfidfs = []
        for elem in tfs:
            tfidfs.append([round(t * i, 3) for t, i in zip(elem, idfs)])
        return tfidfs


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.tfidf_ = TfidfTransformer()

    def fit_transform(self, text):
        count_matrix = super().fit_transform(text)
        return self.tfidf_.fit_transform(count_matrix)


corpus = [
 'Crock Pot Pasta Never boil pasta again',
 'Pasta Pomodoro Fresh ingredients Parmesan to taste'
]

if __name__ == '__main__':
    vectorizer = CountVectorizer()
    print(vectorizer.fit_transform(corpus))
    print(vectorizer.get_feature_names())
    print(tf_transform(vectorizer.fit_transform(corpus)))
    print(idf_transform(vectorizer.fit_transform(corpus)))
    transformer = TfidfTransformer()
    print(transformer.fit_transform(vectorizer.fit_transform(corpus)))
    vectorizer2 = TfidfVectorizer()
    print(vectorizer2.fit_transform(corpus))
    print(vectorizer2.get_feature_names())
