import re
from underthesea import sent_tokenize, word_tokenize, pos_tag

# Dán các danh sách positive_words và negative_words của bạn vào đây
positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "ổn", "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "nhanh", "thân thiện", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp", "an tâm", "thúc đẩy", "cảm động", "nổi trội", "sáng tạo", "phù hợp", "tận tâm", "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận", "vui vẻ", "hào hứng", "đam mê", 'chuyên', 'cảm', 'dễ', 'giỏi', 'hay', 'hiệu', 'hài', 'hỗ trợ', 'nhiệt tình', 'sáng tạo', 'thân', 'thích', 'tuyệt', 'tốt', 'vui', 'ổn','hài lòng', 'chuyên nghiệp', 'động lực', 'dễ chịu', 'xuất sắc', 'công bằng', 'chất lượng', 'ưng ý', 'hoàn hảo', 'hoà nhã', 'tốt', 'hạnh phúc', 'hợp lý', 'thân', 'phù hợp', 'rất thích', 'thân thiện', 'sáng tạo', 'thúc đẩy', 'ổn', 'chăm chỉ', 'truyền cảm hứng', 'phát triển', 'tận tâm', 'cảm động', 'vui', 'nổi bật', 'hợp tác', 'đồng đội', 'hào hứng', 'hòa đồng', 'vui vẻ', 'hiếm có', 'học hỏi', 'tôn trọng', 'tốt nhất', 'vui mừng', 'hiệu', 'tuyệt', 'nhiệt tình', 'thích', 'đẳng cấp', 'dễ dàng', 'chủ động', 'cải thiện', 'đồng cảm', 'cảm', 'dễ', 'mở rộng', 'hỗ trợ', 'bình đẳng', 'chuyên', 'năng động', 'thoải mái', 'mến', 'cảm ơn', 'tốt hơn', 'an tâm', 'tuyệt vời', 'đam mê', 'nhanh', 'giỏi', 'hay', 'hài', 'đáng tin cậy', 'cẩn thận', 'cởi mở', 'cơ hội', 'nổi trội', 'rất tốt'
]
negative_words = [
    "kém", "tệ", "buồn", "chán", "không dễ chịu", "không thích", "không ổn", "áp lực", "chán", "mệt", "không hợp", "không đáng tin cậy", "không chuyên nghiệp", "không thân thiện", "không tốt", "chậm", "khó khăn", "phức tạp", "khó chịu", "gây khó dễ", "rườm rà", "tồi tệ", "khó xử", "không thể chấp nhận", "không rõ ràng", "rối rắm", 'không hài lòng', 'không đáng', 'quá tệ', 'rất tệ', "phiền phức", 'thất vọng', 'tệ hại', 'kinh khủng', 'chán', 'drama', 'dramas', 'gáp', 'gắt', 'kém', 'lỗi', 'mệt', 'ngắt', 'quái', 'quát', 'rối', 'thiếu', 'trễ', 'tệ', 'tệp', 'tồi', 'áp', 'đáp', "hách dịch", 'thất vọng', 'kinh khủng', 'không hài lòng', 'căng thẳng', 'không chuyên nghiệp', 'không hòa đồng', 'quá tệ', 'thiếu đào tạo', 'thiếu sáng tạo', 'gắt', 'không tốt', 'khủng hoảng', 'tệ', 'tồi tệ', 'quát', 'rối loạn', 'khó xử', 'gáp', 'không có cơ hội', 'thiếu công bằng', 'không đáng tin cậy', 'tồi', 'không chấp nhận được', 'không đủ', 'thiếu sự công nhận', 'rối rắm', 'khó chịu', 'buồn', 'thiếu hỗ trợ', 'không hợp', 'dramas', 'mệt', 'thiếu cơ hội thăng tiến', 'trì trệ', 'thất bại', 'thiếu sự minh bạch', 'ngắt', 'chán', 'không rõ ràng', 'không ổn', 'buồn bã', 'chậm', 'rối', 'không đáng', 'mâu thuẫn', 'áp lực', 'quái', 'gây khó dễ', 'thiếu chuyên nghiệp', 'không thể chấp nhận', 'thiếu động lực', 'lo lắng', 'môi trường thiếu cởi mở', 'rườm rà', 'khó khăn', 'thiếu', 'mệt mỏi', 'thiếu linh hoạt', 'tệp', 'trễ', 'không thích', 'phiền phức', 'lỗi', 'không dễ chịu', 'không tôn trọng', 'tức giận', 'tệ hại', 'không phát triển', 'thiếu sự rõ ràng', 'bực bội', 'phức tạp', 'rất tệ', 'hách dịch', 'đáp', 'drama', 'kém', 'không thân thiện'
]


class VietnamesePreprocessor:
    def __init__(self, files_dir=''):
        # Các hàm load file của bạn
        # self.emoji_dict = self._load_dict(files_dir + 'emojicon.txt')
        # ... (các hàm load khác)
        self.emoji_dict = self._load_dict(files_dir + 'emojicon.txt')
        self.teen_dict = self._load_dict(files_dir + 'teencode.txt')
        self.wrong_lst = self._load_list(files_dir + 'wrong-word.txt')
        self.stopwords_lst = self._load_list(files_dir + 'vietnamese-stopwords.txt')
        self.english_dict = self._load_dict(files_dir + 'english-vnmese.txt')


        self.negative_words = set(negative_words)
        negative_words_extend = {'_'.join(word.split()) for word in self.negative_words if ' ' in word}
        self.negative_words.update(negative_words_extend)

        self.positive_words = set(positive_words)
        positive_words_extend = {'_'.join(word.split()) for word in self.positive_words if ' ' in word}
        self.positive_words.update(positive_words_extend)

    # Các hàm _load_dict, _load_list, process_text, loaddicchar, covert_unicode, normalize_repeated_characters giữ nguyên
    def _load_dict(self, filepath):
        d = {}
        try:
            with open(filepath, 'r', encoding='utf8') as f:
                for line in f:
                    if '\t' in line:
                        key, value = line.strip().split('\t')
                        d[key] = value
        except FileNotFoundError:
            print(f"Warning: Dictionary file not found at {filepath}")
            pass
        return d

    def _load_list(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: List file not found at {filepath}")
            return []

    def process_text(self, text):
        document = text.lower()
        document = document.replace('công ty', '')
        document = document.replace("’",'')
        document = re.sub(r'\.+', ".", document)
        new_sentence =''
        # Giả sử sent_tokenize đã được import
        for sentence in sent_tokenize(document):
            sentence = ''.join(self.emoji_dict.get(word, word)+' ' if word in self.emoji_dict else word for word in list(sentence))
            sentence = ' '.join(self.teen_dict.get(word, word) for word in sentence.split())
            pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựỳýỷỹỵđ]+\b'
            sentence = ' '.join(re.findall(pattern,sentence))
            sentence = ' '.join('' if word in self.wrong_lst else word for word in sentence.split())
            new_sentence = new_sentence+ sentence + '. '
        document = new_sentence
        document = re.sub(r'\s+', ' ', document).strip()
        return document

    def loaddicchar(self):
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
        dic = {char1252[i]: charutf8[i] for i in range(len(char1252))}
        return dic

    def covert_unicode(self, txt):
        dicchar = self.loaddicchar()
        return re.sub(
            r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: dicchar[x.group()], txt)

    def normalize_repeated_characters(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    # PHIÊN BẢN SỬA LỖI CUỐI CÙNG
    # (Không còn các lệnh print debug)

    # PHIÊN BẢN HOÀN CHỈNH
    # Kết hợp Word Segmentation và POS Tagging

    def process_postag_thesea(self, text):
        new_document_words = []
        for sentence in sent_tokenize(text):
            sentence = sentence.replace('.', '').strip()
            if not sentence:
                continue

            try:
                # BƯỚC 1: Dùng word_tokenize để nối các cụm từ trước.
                # Kết quả là một string duy nhất, ví dụ: 'cơ_sở vật_chất đẹp'
                segmented_sentence = word_tokenize(sentence, format="text")

                # BƯỚC 2: Đưa string đã được nối từ vào pos_tag.
                # Vì đầu vào là string nên sẽ không bị lỗi.
                # pos_tag sẽ gán nhãn cho các token đã được nối.
                tagged_words = pos_tag(segmented_sentence)

            except Exception as e:
                # print(f"Lỗi khi xử lý câu '{sentence}': {e}") # Có thể mở để debug nếu cần
                continue

            # Logic xử lý từ "không" (giữ nguyên, đã đúng)
            processed_words = []
            i = 0
            while i < len(tagged_words):
                word, tag = tagged_words[i]
                if word == 'không':
                    if i + 2 < len(tagged_words):
                        next_word_1, next_word_2 = tagged_words[i+1][0], tagged_words[i+2][0]
                        phrase_3 = f"không_{next_word_1}_{next_word_2}"
                        if phrase_3 in self.negative_words:
                            processed_words.append((phrase_3, 'A'))
                            i += 3
                            continue
                    if i + 1 < len(tagged_words):
                        next_word_1 = tagged_words[i+1][0]
                        phrase_2 = f"không_{next_word_1}"
                        if phrase_2 in self.negative_words:
                            processed_words.append((phrase_2, 'A'))
                            i += 2
                            continue
                processed_words.append((word, tag))
                i += 1

            # Lọc từ theo từ loại (giữ nguyên)
            lst_word_type = ['N', 'NP', 'NC', 'NY', 'A', 'AB', 'AJ', 'V', 'VB', 'VY', 'R']
            final_words = [word for word, tag in processed_words if tag.upper() in lst_word_type]
            new_document_words.extend(final_words)

        return ' '.join(new_document_words)

    def remove_stopword(self, text):
        document = ' '.join('' if word in self.stopwords_lst else word for word in text.split())
        document = re.sub(r'\s+', ' ', document).strip()
        return document

    def translate_english_to_vietnamese(self, text):
        words = text.split()
        translated_words = [self.english_dict.get(word, word) for word in words]
        return ' '.join(translated_words)


    def full_preprocess(self, text):
        text = self.process_text(text)
        text = self.covert_unicode(text)
        text = self.translate_english_to_vietnamese(text) # Add translation step
        text = self.normalize_repeated_characters(text)
        text = self.process_postag_thesea(text)
        text = self.remove_stopword(text)
        return text

    @staticmethod
    def find_words(document, list_of_words):
        document_lower = document.lower()
        word_count = 0
        word_list = []
        for word in list_of_words:
            if word in document_lower:
                word_count += document_lower.count(word)
                word_list.append(word)
        return word_count, word_list